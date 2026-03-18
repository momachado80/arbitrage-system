# Audit: Diagnóstico de Funil de Elegibilidade Estrutural

## 1. Hipótese testada

Depois de derrotar a hipótese de congelamento operacional estrutural (problema era semântico de observabilidade com `lastUpdate`), o próximo passo é descobrir **onde a família estrutural está morrendo economicamente** no fluxo. Este diagnóstico implementa um funil de elegibilidade por profile para separar claramente cada etapa de filtragem e identificar o principal choke point.

## 2. Por que este é o próximo passo correto

- **lastCycleProcessedAt** já garante que sabemos quando cada profile foi processado.
- O problema restante é **causal**: em que etapa do fluxo (pair → fill → capfloor → degratio → open → close) a família estrutural perde viabilidade?
- Sem instrumentação por etapa, não é possível distinguir entre:
  1. Profile processado, mas sem oportunidades brutas
  2. Profile com oportunidades brutas, mas morto em pair/fill
  3. Profile passando pair/fill, mas morto em capfloor/degratio
  4. Profile chegando a candidate, mas não abrindo
  5. Profile abrindo, mas destruindo economia no close

O funil de elegibilidade resolve isso com contadores cumulativos por profile e razões agregadas de descarte.

## 3. Campos adicionados

### A. Instrumentação no runtime (`profileEligibilityFunnel.ts`)

| Campo | Descrição |
|-------|-----------|
| `cyclesProcessed` | Ciclos em que o profile foi processado |
| `rawOpportunitiesSeen` | Oportunidades brutas avaliadas (profile × opp) |
| `pairEligibleCount` | Passou no filtro de pair set (structural risk) |
| `fillEligibleCount` | Passou no filtro de fill bucket |
| `capfloorEligibleCount` | Passou no filtro de capfloor (≥ 4.5%) |
| `degratioEligibleCount` | Passou no filtro de degratio (≥ 0.24) |
| `finalCandidateCount` | Passou todos os gates, pré-addShadowTrade |
| `openAttemptCount` | Tentativas de abertura (igual a finalCandidate) |
| `openedTradeCount` | Trades efetivamente abertos |
| `closedTradeCount` | Trades fechados (do store) |
| `realizedPnlTotal` | PnL realizado agregado |
| `realizedPnlAvg` | PnL médio por close |

`discardReasonCounts` vem de `getRejectionCountsByProfile()` (store existente) — razões agregadas de descarte por gate.

### B. Bloco no audit

`profileEligibilityDiagnostics` — por profile:
- `profileId`, `label`
- `cyclesProcessed`, `rawOpportunitiesSeen`
- `funnel` (pair, fill, capfloor, degratio, finalCandidate, openAttempt, opened, closed)
- `discardReasonCounts`
- `conversions` (rawToPair, pairToFill, fillToCapfloor, etc.)
- `rates` (openRate, closeRate)
- `economics` (realizedPnlTotal, realizedPnlAvg)
- `chokePointSummary` (leitura textual curta do principal choke)

### C. Endpoint profiles

Cada profile inclui `eligibilityFunnel`:
- `cyclesProcessed`, `rawOpportunitiesSeen`, `finalCandidateCount`, `openedTradeCount`, `closedTradeCount`
- `realizedPnlTotal`, `realizedPnlAvg`
- `chokePointSummary`

## 4. Como interpretar o funil

1. **Profile processado, mas sem oportunidades brutas**  
   - `cyclesProcessed` > 0, `rawOpportunitiesSeen` = 0  
   - Choke: upstream (merge, graph, standard) não está entregando opps para este profile

2. **Oportunidades brutas, morto em pair/fill**  
   - `rawOpportunitiesSeen` > 0, `pairEligibleCount` = 0 ou `fillEligibleCount` = 0  
   - `discardReasonCounts` com `structural_risk_pair_mismatch` ou `structural_risk_fill_bucket_mismatch` dominante

3. **Passa pair/fill, morto em capfloor/degratio**  
   - `fillEligibleCount` > 0, `capfloorEligibleCount` = 0 ou `degratioEligibleCount` = 0  
   - `structural_risk_capfloor` ou `structural_risk_degratio` em destaque

4. **Chega a candidate, não abre**  
   - `finalCandidateCount` > 0, `openedTradeCount` = 0  
   - Indica possível bug (addShadowTrade deveria ser chamado)

5. **Abre, mas destrói economia no close**  
   - `openedTradeCount` > 0, `closedTradeCount` > 0, `realizedPnlTotal` < 0  
   - Choke: lógica de exit ou condições de mercado

## 5. Comandos para validar em produção

```bash
# Audit completo (inclui profileEligibilityDiagnostics)
curl -s "https://<DASHBOARD_URL>/api/shadow/audit" | jq '.profileEligibilityDiagnostics'

# Apenas perfis estruturais (structural risk managed)
curl -s "https://<DASHBOARD_URL>/api/shadow/audit" | jq '.profileEligibilityDiagnostics["shadow_1000_structural_riskmanaged_v1"]'

# Profiles com funil resumido
curl -s "https://<DASHBOARD_URL>/api/shadow/profiles" | jq '.profiles[] | {profileId, eligibilityFunnel}'

# Choke point por profile
curl -s "https://<DASHBOARD_URL>/api/shadow/audit" | jq '.profileEligibilityDiagnostics | to_entries[] | {profileId: .key, chokePointSummary: .value.chokePointSummary}'

# Discard reasons do profile estrutural
curl -s "https://<DASHBOARD_URL>/api/shadow/audit" | jq '.profileEligibilityDiagnostics["shadow_1000_structural_riskmanaged_v1"].discardReasonCounts'
```

## 6. Restrições respeitadas

- ✅ Não criar novo challenger
- ✅ Não alterar lógica econômica dos profiles
- ✅ Apenas observabilidade e diagnósticos
- ✅ Compatibilidade com a esteira atual mantida
