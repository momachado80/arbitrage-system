# Structural Fillrelax + Capfloor-Relax Challenger — Branch C

## 1. Profile pai

`shadow_1000_structural_fillrelax_v1` — Structural fill-relax (pair×fill 0.1–0.5×capfloor×degratio).

Evidência em produção: fillrelax removeu o choke em fill com sucesso. O choke dominante migrou para capfloor (fillToCapfloor ≈ 0.016; capfloorEligibleCount = 2 vs fillEligibleCount = 123).

## 2. Única mudança causal

**capfloor:** `4.5%` (0.045) → `3%` (0.03)

O pai exige `capturableEdgeAtEntry >= 4.5%`. O challenger relaxa o piso para 3%, aceitando opps com edge entre 3% e 4.5% que antes eram rejeitadas. Demais gates (pair set, fill 0.1–0.5, degratio 0.24, exit, sizing adaptativo) permanecem idênticos.

## 3. Hipótese testada

Depois de remover o choke em fill, o capfloor passou a ser o gargalo dominante. Um relaxamento controlado do capfloor (4.5% → 3%) deve aumentar `capfloorEligibleCount`, `finalCandidateCount` e `openedTradeCount` sem degradar qualidade econômica.

## 4. Critério de sucesso mecânico

- `capfloorEligibleCount` > 0 e maior que o pai (fillrelax)
- `fillToCapfloor` (conversão fill→capfloor) maior que o pai
- `finalCandidateCount` e `openedTradeCount` crescentes ao longo do tempo

## 5. Critério de sucesso econômico

- Após amostra suficiente de closes (≥ 20): `realizedPnLTotal` e `realizedPnlAvg` não piores que o pai (fillrelax), preferencialmente melhores
- Win rate e loss magnitude consistentes ou melhores que fillrelax

## 6. O que não concluir cedo demais

- **Não** concluir sucesso com poucos closes (ex.: < 20)
- **Não** comparar PnL antes de amostra equivalente de ciclos
- **Não** alterar exit, fill, pair, degratio ou outros gates; isolar apenas o efeito do capfloor
- **Não** empilhar outros challengers ou mudanças em paralelo

## Profile ID

`shadow_1000_structural_fillrelax_capfloorrelax_v1`

## Comandos de validação

```bash
curl -s "https://dashboard-next-production-b126.up.railway.app/api/shadow/audit" | jq '.profileEligibilityDiagnostics["shadow_1000_structural_fillrelax_capfloorrelax_v1"]'
curl -s "https://dashboard-next-production-b126.up.railway.app/api/shadow/profiles" | jq '.profiles[] | select(.profileId == "shadow_1000_structural_fillrelax_capfloorrelax_v1")'

# Comparar pai vs challenger
curl -s "https://dashboard-next-production-b126.up.railway.app/api/shadow/audit" | jq '{
  fillrelax: .profileEligibilityDiagnostics["shadow_1000_structural_fillrelax_v1"].funnel,
  capfloorrelax: .profileEligibilityDiagnostics["shadow_1000_structural_fillrelax_capfloorrelax_v1"].funnel
}'
```
