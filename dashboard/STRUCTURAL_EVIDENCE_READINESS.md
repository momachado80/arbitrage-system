# Structural Evidence Readiness — Legibilidade no Ramo A

## 1. Por que este passo é o correto no ramo A

Com `evidenceSufficient = false`, a decisão é `do_not_modify_yet`. O sistema já entrega essa conclusão via `structuralRecalibrationReview`, mas não explicita **quanto falta** para sair do ramo A. O `structuralEvidenceReadiness` adiciona essa legibilidade sem alterar economia, profiles, execução ou thresholds. É puramente observacional.

## 2. O que significa readiness

- **NOT_READY:** Ao menos um critério (ciclos, opps, profiles) está abaixo do mínimo. Manter coleta.
- **NEAR_READY:** Restam poucos ciclos (≤3) para o mínimo, e opps/profiles já são suficientes. Atenção para rechecagem em breve.
- **READY:** Todos os critérios atingidos. `evidenceSufficient = true`. Autorizado a avaliar ramo B ou C na próxima análise.

## 3. Por que sinal provisório não é decisão

O `provisionalSignal` (ex: `fill_gate_preview`) reflete o padrão atual nos `chokePointSummary` dos profiles estruturais. É útil para antecipar o que pode ser priorizado quando houver evidência, mas:

- Não muda `evidenceSufficient`
- Não altera thresholds
- Não dispara recalibração
- `provisionalSignalIsDecisive` é sempre `false`

O `nextDecisionBranchWhenReady` indica a rota provável se o padrão se manter, mas a decisão real só ocorre após `evidenceSufficient = true`.

## 4. Quando estaremos autorizados a sair do `do_not_modify_yet`

Quando `structuralEvidenceReadiness.readinessStatus = "READY"` (e portanto `evidenceSufficient = true`):

1. Ler novamente o `structuralRecalibrationReview`
2. Seguir a árvore de decisão:
   - Se `recommendedNextSingleHypothesis.action = "inspect_exit_economics"` → Ramo B
   - Se houver gate pré-open dominante → Ramo C
   - Caso contrário, manter monitoramento

## 5. Comandos de validação em produção

```bash
DASHBOARD_URL="https://dashboard-next-production-b126.up.railway.app"

# Readiness completo
curl -s "${DASHBOARD_URL}/api/shadow/audit" | jq '.structuralEvidenceReadiness'

# Apenas quanto falta
curl -s "${DASHBOARD_URL}/api/shadow/audit" | jq '.structuralEvidenceReadiness | {readinessStatus, cyclesRemaining, rawOpportunitiesRemaining, profilesRemaining, provisionalSignal, summary}'

# Rechecar quando READY
curl -s "${DASHBOARD_URL}/api/shadow/audit" | jq '.structuralEvidenceReadiness.readinessStatus'
```
