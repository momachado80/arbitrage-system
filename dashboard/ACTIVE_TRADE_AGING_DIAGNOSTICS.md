# Active Trade Aging Diagnostics

## Objetivo

Esclarecer o comportamento de trades ativos que permanecem abertos sem close, especialmente quando um profile tem poucos opens e zero closes após horas de deploy (ex.: `shadow_1000_structural_fillrelax_capfloorrelax_v1` com 2 opens e 0 closes).

## Bloco `activeTradeAgingDiagnostics` em `/api/shadow/audit`

Por profile:

- **profileId**, **label**
- **activeTradesCount**
- **activeTradesAging**: array com detalhes por trade ativo
- **oldestActiveTradeMs**, **avgActiveTradeMs**
- **activeTradeIds**
- **avgObservedEdgeCurrent**, **avgNetEdgeCurrent**
- **likelyExitReasonIfClosedNow**
- **nearMaxHoldingCount**
- **summary**: diagnóstico textual

Por trade ativo:

- **openedAt**, **holdingMs**
- **entryEdge** (observedEdgeAtEntry)
- **currentObservedEdge**, **currentNetEdge**
- **currentPnlPct**
- **likelyExitReasonIfClosedNow**
- **nearMaxHolding**

## Categorias de summary

1. **Trades ativos ainda dentro do regime normal** — holding &lt; 30% do max holding
2. **Trades envelhecidos e presos** — nearMaxHoldingCount &gt; 0 e oldest &gt; 90% do max holding
3. **Oportunidade ausente no merge atual** — opportunityId não encontrado; possível opp desapareceu ou anomalia
4. **Baixa geração de novos candidates** — finalCandidateCount baixo; poucos candidates após os opens
5. **Possível anomalia operacional** — combinação de trades longos, zero closes e poucos candidates

## Exit reasons inferidos

- **max_holding_time**: holdingMs >= maxHoldingTimeMs
- **stop_loss**: pnlPct <= -stopLossPct
- **take_profit**: pnlPct >= takeProfitPct
- **edge_normalization**: |currentEdge| &lt; 0.005
- **opportunity_absent_or_unknown**: opp não está no merge atual
- **within_hold_no_exit_trigger**: nenhum gatilho atingido

## Comandos de validação

```bash
# Bloco completo
curl -s "https://dashboard-next-production-b126.up.railway.app/api/shadow/audit" | jq '.activeTradeAgingDiagnostics'

# Capfloorrelax específico
curl -s "https://dashboard-next-production-b126.up.railway.app/api/shadow/audit" | jq '.activeTradeAgingDiagnostics.byProfile["shadow_1000_structural_fillrelax_capfloorrelax_v1"]'

# Por trade ativo
curl -s "https://dashboard-next-production-b126.up.railway.app/api/shadow/audit" | jq '.activeTradeAgingDiagnostics.byProfile["shadow_1000_structural_fillrelax_capfloorrelax_v1"].activeTradesAging'
```
