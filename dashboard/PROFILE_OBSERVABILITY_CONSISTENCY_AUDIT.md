# Profile Observability Consistency Audit

## 1. Causa da inconsistência

A inconsistência observada — profile exibindo `chokePointSummary = "profile não processado"` enquanto possui `closedTradeCount > 0` e `realizedPnlTotal ≠ 0` — ocorre porque:

- **Contadores de funil** (`cyclesProcessed`, `rawOpportunitiesSeen`, `pairEligibleCount`, etc.) são **voláteis**: ficam em memória e são zerados a cada **deploy/restart** do dashboard.
- **Histórico econômico** (`closedTrades`, `realizedPnL`) é **persistido** (file persistence) e **reidratado** na inicialização.
- Após um deploy, o pai (`shadow_1000_structural_riskmanaged_v1`) tem 327 closes e PnL acumulado vindos da reidratação, mas `cyclesProcessed === 0` porque o runtime atual ainda não processou ciclos.
- A lógica antiga de `chokePointSummary` priorizava `cyclesProcessed === 0` e retornava "profile não processado", ignorando evidência econômica reidratada.

## 2. Impacto na leitura pai vs challenger

- **Pai (riskmanaged):** Aparecia como "não processado" mesmo com 327 closes e PnL — leitura enganosa.
- **Challenger (fillrelax):** Tudo zerado pode significar:
  - Profile novo sem runtime suficiente (esperado); ou
  - Profile desabilitado/faltando materialização; ou
  - Realmente sem dados ainda.
- A comparação pai vs challenger ficava ilegível: o pai parecia "não processado" apesar de economia rica, enquanto o challenger zerado podia ser apenas novo.

## 3. O que já pode ser concluído

Com o bloco `profileObservabilityConsistency` e o `chokePointSummary` corrigido:

- **FUNNEL_MISSING_BUT_ECONOMICS_PRESENT**: Indica claramente "histórico econômico reidratado presente; contadores de funil zerados (reset pós-deploy)". A decisão operacional é: aguardar ciclos pós-deploy para re-materializar o funil antes de comparar.
- **NEW_PROFILE_NO_DATA_YET**: Profile novo ou sem runtime suficiente — aguardar ciclos.
- **HEARTBEAT_PRESENT_BUT_FUNNEL_EMPTY**: Heartbeat ativo mas funnel vazio — possível reset ou falta de opps matching.
- **CONSISTENT**: Funil, economia e heartbeat coerentes — comparação confiável.

## 4. O que ainda não pode ser concluído

- **Julgamento do fillrelax:** Se o challenger está em `NEW_PROFILE_NO_DATA_YET` ou com funnel zerado, **ainda não é possível** concluir se o relaxamento do fill bucket melhora ou não a economia. É necessário:
  - Ciclos suficientes após deploy para materializar `pairEligibleCount`, `fillEligibleCount`, etc.
  - Closes suficientes para PnL significativo.
- **Comparação direta PnL pai vs challenger:** Só válida quando ambos têm `consistencyStatus = CONSISTENT` ou quando se entende que o pai está em `FUNNEL_MISSING_BUT_ECONOMICS_PRESENT` (PnL reidratado é histórico, não deste runtime).

## 5. Comandos de validação em produção

```bash
# Bloco profileObservabilityConsistency (por profile)
curl -s "https://dashboard-next-production-b126.up.railway.app/api/shadow/audit" | jq '.profileObservabilityConsistency'

# Consistência dos perfis pai e challenger
curl -s "https://dashboard-next-production-b126.up.railway.app/api/shadow/audit" | jq '.profileObservabilityConsistency.byProfile["shadow_1000_structural_riskmanaged_v1"]'
curl -s "https://dashboard-next-production-b126.up.railway.app/api/shadow/audit" | jq '.profileObservabilityConsistency.byProfile["shadow_1000_structural_fillrelax_v1"]'

# chokePointSummary (deve refletir economia quando existir)
curl -s "https://dashboard-next-production-b126.up.railway.app/api/shadow/audit" | jq '.profileEligibilityDiagnostics["shadow_1000_structural_riskmanaged_v1"].chokePointSummary'
curl -s "https://dashboard-next-production-b126.up.railway.app/api/shadow/audit" | jq '.profileEligibilityDiagnostics["shadow_1000_structural_fillrelax_v1"].chokePointSummary'

# Diagnóstico full do pai e challenger
curl -s "https://dashboard-next-production-b126.up.railway.app/api/shadow/audit" | jq '.profileEligibilityDiagnostics["shadow_1000_structural_riskmanaged_v1"]'
curl -s "https://dashboard-next-production-b126.up.railway.app/api/shadow/audit" | jq '.profileEligibilityDiagnostics["shadow_1000_structural_fillrelax_v1"]'
```

## Decisão operacional sobre o fillrelax

- Se `profileObservabilityConsistency.byProfile["shadow_1000_structural_fillrelax_v1"].consistencyStatus` for `NEW_PROFILE_NO_DATA_YET` ou `INCONCLUSIVE`: **ainda não dá para julgar**. Aguardar ciclos e revalidar.
- Se, após ciclos suficientes, o fillrelax tiver `fillEligibleCount > 0` e closes acumulados, a comparação PnL com o pai passa a ser legível.
- O `profileObservabilityConsistency` indica explicitamente quando a leitura é confiável vs quando aguardar.
