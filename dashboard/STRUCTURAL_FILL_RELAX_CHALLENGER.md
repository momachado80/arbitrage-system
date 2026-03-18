# Structural Fill-Relax Challenger — Branch B

## Profile pai

`shadow_1000_structural_riskmanaged_v1` — Structural risk-managed pair×fill×capfloor×degratio.

## Única mudança causal

**fillRatioBucket:** `0.1-0.25` → `0.1-0.5`

O pai aceita apenas fill ratio entre 10% e 25%. O challenger relaxa o teto para 50%, mantendo o piso em 10%. Demais gates (pair set, capfloor 4.5%, degratio 0.24, sizing adaptativo) permanecem idênticos.

## Hipótese econômica

O `structuralRecalibrationReview` identificou **fill_gate** como choke dominante: 5 profiles estruturais estrangulados em fill (pair passa, fill mata). A hipótese é que o bucket 0.1–0.25 está excluindo opps economicamente viáveis com fill entre 25% e 50%. O challenger testa se expandir o bucket aumenta candidates e opens sem degradar qualidade.

## Evidência de sucesso

- `fillEligibleCount` > 0 para o challenger
- `finalCandidateCount` e `openedTradeCount` crescentes ao longo do tempo
- Após closes suficientes: `realizedPnLTotal` e `realizedPnlAvg` não piores que o pai, preferencialmente melhores

## O que não concluir cedo demais

- **Não** concluir sucesso com poucos closes (ex.: < 10)
- **Não** comparar PnL antes de amostra equivalente de ciclos
- **Não** alterar exit ou outros gates; isolar apenas o efeito do fill bucket
- **Não** empilhar outros challengers ou mudanças em paralelo

## Profile ID

`shadow_1000_structural_fillrelax_v1`

## Comandos de validação

```bash
curl -s "https://dashboard-next-production-b126.up.railway.app/api/shadow/audit" | jq '.profileEligibilityDiagnostics["shadow_1000_structural_fillrelax_v1"]'
curl -s "https://dashboard-next-production-b126.up.railway.app/api/shadow/profiles" | jq '.profiles[] | select(.profileId == "shadow_1000_structural_fillrelax_v1")'
```
