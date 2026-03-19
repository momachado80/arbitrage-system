# Structural Fillrelax + Capfloor-Relax + Degratio-Relax Challenger — Branch D

## 1. Profile pai

`shadow_1000_structural_fillrelax_capfloorrelax_v1` — Structural fill+capfloor-relax (pair×fill 0.1–0.5×capfloor 3%×degratio 0.24).

Evidência em produção: fillrelax e capfloorrelax destravaram fill e capfloor com sucesso. O choke dominante migrou para degratio (capfloorEligibleCount = 2, degratioEligibleCount = 0, finalCandidateCount = 0, openedTradeCount = 0).

## 2. Única mudança causal

**degratio:** `degRatioMin 0.24` → `0.20`

O pai exige `capturable/observed >= 0.24`. O challenger relaxa o piso para 0.20, aceitando opps com degratio entre 0.20 e 0.24 que antes eram rejeitadas. Demais gates (pair set, fill 0.1–0.5, capfloor 3%, exit, sizing adaptativo) permanecem idênticos.

## 3. Hipótese testada

Depois de remover os gargalos de fill e capfloor, o degratio se tornou o choke dominante. Um relaxamento controlado do degratio (0.24 → 0.20) deve aumentar `degratioEligibleCount`, `finalCandidateCount` e `openedTradeCount` sem degradar qualidade econômica.

## 4. Critério de sucesso mecânico

- `degratioEligibleCount` > 0 e maior que o pai (capfloorrelax)
- `capfloorToDegratio` (conversão capfloor→degratio) maior que o pai
- `finalCandidateCount` e `openedTradeCount` crescentes ao longo do tempo

## 5. Critério de sucesso econômico

- Após amostra suficiente de closes (≥ 20): `realizedPnLTotal` e `realizedPnlAvg` não piores que o pai (capfloorrelax), preferencialmente melhores
- Win rate e loss magnitude consistentes ou melhores que capfloorrelax

## 6. O que não concluir cedo demais

- **Não** concluir sucesso com poucos closes (ex.: < 20)
- **Não** comparar PnL antes de amostra equivalente de ciclos
- **Não** alterar exit, fill, pair, capfloor ou outros gates; isolar apenas o efeito do degratio
- **Não** empilhar outros challengers ou mudanças em paralelo

## Profile ID

`shadow_1000_structural_fillrelax_capfloorrelax_degratiorelax_v1`

## Comandos de validação

```bash
# Snapshot do degratiorelax
curl -s "https://<DASHBOARD_URL>/api/shadow/profile-snapshot?profileId=shadow_1000_structural_fillrelax_capfloorrelax_degratiorelax_v1" | jq .

# Comparar pai vs challenger (funnel)
curl -s "https://<DASHBOARD_URL>/api/shadow/audit" | jq '{
  capfloorrelax: .profileEligibilityDiagnostics["shadow_1000_structural_fillrelax_capfloorrelax_v1"].funnel,
  degratiorelax: .profileEligibilityDiagnostics["shadow_1000_structural_fillrelax_capfloorrelax_degratiorelax_v1"].funnel
}'

# Profile no audit
curl -s "https://<DASHBOARD_URL>/api/shadow/audit" | jq '.profileAtomicSnapshot.byProfile["shadow_1000_structural_fillrelax_capfloorrelax_degratiorelax_v1"]'
```
