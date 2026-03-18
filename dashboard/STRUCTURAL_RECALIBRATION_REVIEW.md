# Review: Shortlist Objetiva de Recalibração Estrutural

## 1. Objetivo do review

Consolidar os diagnósticos dos profiles estruturais (`profileEligibilityDiagnostics` + `profileEligibilityJudgements`) e produzir uma **shortlist ordenada** de choke points/gates candidatos a recalibração, com **uma única hipótese prioritária** recomendada. Sem alterar lógica econômica; apenas prioridade de ação.

## 2. Critérios usados

### Evidência suficiente

O review bloqueia recomendações de recalibração quando:

- Menos de 2 profiles estruturais materializados
- Menos de 50 opps brutas agregadas
- Menos de 10 ciclos processados

Nesses casos: `evidenceSufficient: false`, `blockedRecalibrations` inclui todos os gates, `recommendedNextSingleHypothesis.action = "keep_collecting"`.

### Post-open dominante

Se o choke `post_open_negative_economics` afeta a maioria dos profiles e a pressão pós-open ≥ pré-open:

- **Não recalibrar entrada.** O problema está no close, não nos gates de entrada.
- `recalibrationCandidates = []`, `blockedRecalibrations` = todos.
- `recommendedNextSingleHypothesis.action = "inspect_exit_economics"`.

### Pre-open dominante

Se um gate pré-open (pair, fill, capfloor, degratio, candidate_to_open) domina e há evidência MODERATE ou STRONG:

- Priorizar esse gate como única hipótese.
- `recalibrationCandidates = [gate]`, demais gates em `blockedRecalibrations`.
- `recommendedNextSingleHypothesis.gate` = gate prioritário.

### Uma única hipótese

O review **nunca** recomenda múltiplas recalibrações ao mesmo tempo.

## 3. Como interpretar o ranking de chokes

`gatePressureRanking` ordena os gates por:

1. `affectedProfilesCount` (maior primeiro)
2. `totalRawOpportunitiesSeen` (desempate)

Para cada gate:

- `affectedProfilesCount`: quantos profiles estruturais têm esse choke dominante
- `chokeFrequency`: fração de profiles revisados afetados
- `relativePressure`: peso relativo entre todos os chokes
- `supportingProfiles`: lista de profileIds com esse choke
- `evidenceGrade`: WEAK | MODERATE | STRONG

`dominantStructuralChokesRanked` é a ordem de prioridade por pressão.

## 4. Como interpretar recommendedNextSingleHypothesis

- **gate**: gate prioritário para recalibração, ou `null` se nenhum.
- **action**: ação concreta (ex: "relaxamento controlado do fill bucket", "inspect_exit_economics", "keep_collecting").
- **reason**: justificativa breve.

Se `gate === null` e `action === "keep_collecting"`: amostra insuficiente.

Se `gate === null` e `action === "inspect_exit_economics"`: problema está no close; não mexer em entrada.

Se `gate` é um gate pré-open: essa é a única recalibração a testar primeiro como challenger causal.

## 5. O que explicitamente não fazer

1. **Não recalibrar múltiplos gates** ao mesmo tempo.
2. **Não recalibrar entrada** quando `post_open_negative_economics` domina.
3. **Não alterar lógica econômica** dos profiles existentes.
4. **Não criar challenger** ainda — o review apenas prioriza; a implementação é separada.
5. **Não ignorar** `evidenceSufficient: false` — quando for false, manter coleta.

## 6. Comandos de validação em produção

```bash
# Review completo
curl -s "https://<DASHBOARD_URL>/api/shadow/audit" | jq '.structuralRecalibrationReview'

# Resumo executivo e hipótese única
curl -s "https://<DASHBOARD_URL>/api/shadow/audit" | jq '.structuralRecalibrationReview | {executiveSummary, recommendedNextSingleHypothesis, evidenceSufficient}'

# Ranking de gates
curl -s "https://<DASHBOARD_URL>/api/shadow/audit" | jq '.structuralRecalibrationReview.gatePressureRanking'

# Breakdown pré vs pós-open
curl -s "https://<DASHBOARD_URL>/api/shadow/audit" | jq '.structuralRecalibrationReview.preOpenVsPostOpenBreakdown'

# Candidatos e bloqueados
curl -s "https://<DASHBOARD_URL>/api/shadow/audit" | jq '.structuralRecalibrationReview | {recalibrationCandidates, blockedRecalibrations}'
```
