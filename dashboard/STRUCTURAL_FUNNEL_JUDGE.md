# Judge: Decisão Operacional a partir do Funil de Elegibilidade

## 1. Objetivo do judge

Transformar os dados do funil de elegibilidade (`profileEligibilityDiagnostics`) em **decisão operacional automatizada**. O judge responde, por profile:

- Onde está o choke dominante?
- Qual o status de saúde do funil?
- Qual ação operacional recomendar?
- Qual o grau de evidência?

Sem alterar regras econômicas; apenas observabilidade → decisão.

## 2. Critérios de classificação

### dominantChokeStage

| Valor | Condição causal |
|-------|-----------------|
| `upstream_no_opportunities` | `cyclesProcessed` alto, `rawOpportunitiesSeen` baixo ou zero |
| `pair_gate` | Opps brutas, mas morre antes de `pairEligibleCount`; `structural_risk_pair_mismatch` dominante |
| `fill_gate` | Passa pair, morre em fill; `structural_risk_fill_bucket_mismatch` dominante |
| `capfloor_gate` | Passa fill, morre em capfloor; `structural_risk_capfloor` dominante |
| `degratio_gate` | Passa capfloor, morre em degratio; `structural_risk_degratio` dominante |
| `candidate_to_open` | `finalCandidateCount` > 0 mas `openedTradeCount` = 0 |
| `post_open_negative_economics` | Abre, fecha, `realizedPnLTotal` < 0 |
| `insufficient_sample` | `cyclesProcessed` < 5 (amostra insuficiente) |
| `no_clear_choke` | Fluxo ativo ou sem choke identificável |

### funnelHealthStatus

| Valor | Significado |
|-------|-------------|
| `HEALTHY_FLOW` | Fluxo ativo, PnL realizado não negativo |
| `STARVED_UPSTREAM` | Sem opps chegando do merge/graph |
| `OVERFILTERED_EARLY` | Choke em pair ou fill (gates iniciais) |
| `OVERFILTERED_LATE` | Choke em capfloor ou degratio |
| `OPENING_BROKEN` | Chega a candidate mas não abre |
| `ECONOMICALLY_NEGATIVE_AFTER_OPEN` | Destrói economia no close |
| `INCONCLUSIVE` | Amostra insuficiente ou sem conclusão |

### evidenceGrade

| Valor | Critério |
|-------|----------|
| `WEAK` | Poucos ciclos, poucos closes ou amostra pequena |
| `MODERATE` | Dados suficientes para opinar, mas não abundantes |
| `STRONG` | Muitos ciclos, muitos opps ou muitos closes |

### recommendedAction

| Valor | Quando usar |
|-------|-------------|
| `keep_collecting` | Amostra insuficiente |
| `inspect_upstream_supply` | Choke em upstream (sem opps) |
| `inspect_pair_set` | Choke em pair |
| `inspect_fill_bucket` | Choke em fill |
| `inspect_capfloor` | Choke em capfloor |
| `inspect_degratio` | Choke em degratio |
| `inspect_open_transition` | Chega a candidate mas não abre |
| `inspect_exit_economics` | PnL negativo após close |
| `eligible_for_recalibration_review` | Principal descarte identificado; revisar entrada |
| `do_not_modify_yet` | Fluxo ok ou inconclusivo; manter monitoramento |

## 3. Como interpretar dominantChokeStage

- **upstream_no_opportunities**: O profile está sendo processado, mas o merge (graph + standard) não está entregando oportunidades. Verificar se outros profiles recebem opps; se sim, pode ser configuração específica.
- **pair_gate / fill_gate / capfloor_gate / degratio_gate**: O profile estrutural está estrangulado em um gate específico. O próximo passo é inspecionar esse gate (conjunto de pares, bucket de fill, thresholds de capfloor/degratio).
- **candidate_to_open**: Lógica de abertura possivelmente quebrada — candidatos existem mas `addShadowTrade` não é chamado.
- **post_open_negative_economics**: A decisão de saída (exit) ou as condições de mercado estão destruindo valor.
- **insufficient_sample / no_clear_choke**: Manter coleta de dados ou monitoramento; não alterar configuração ainda.

## 4. Como interpretar recommendedAction

A ação recomendada é o **próximo passo operacional** sugerido pelo judge:

- `inspect_*` → Investigar aquele componente (supply, pair set, fill bucket, capfloor, degratio, open transition, exit economics).
- `eligible_for_recalibration_review` → Há evidência de que um critério de entrada está filtrando demais; revisar com cautela.
- `keep_collecting` / `do_not_modify_yet` → Não modificar; continuar observando.

## 5. Comandos de validação em produção

```bash
# Julgamentos por profile
curl -s "https://<DASHBOARD_URL>/api/shadow/audit" | jq '.profileEligibilityJudgements'

# Apenas structural risk managed
curl -s "https://<DASHBOARD_URL>/api/shadow/audit" | jq '.profileEligibilityJudgements["shadow_1000_structural_riskmanaged_v1"]'

# Resumo: choke e recomendação por profile
curl -s "https://<DASHBOARD_URL>/api/shadow/audit" | jq '.profileEligibilityJudgements | to_entries[] | {profileId: .key, dominantChokeStage: .value.dominantChokeStage, recommendedAction: .value.recommendedAction, summary: .value.summary}'

# Profiles com saúde do funil
curl -s "https://<DASHBOARD_URL>/api/shadow/audit" | jq '.profileEligibilityJudgements | to_entries[] | {profileId: .key, funnelHealthStatus: .value.funnelHealthStatus, evidenceGrade: .value.evidenceGrade}'

# Profiles que precisam inspeção de exit
curl -s "https://<DASHBOARD_URL>/api/shadow/audit" | jq '.profileEligibilityJudgements | to_entries[] | select(.value.recommendedAction == "inspect_exit_economics") | {profileId: .key, summary: .value.summary}'
```

## 6. Restrições

- Não altera regras econômicas dos profiles.
- Não cria challenger novo.
- Não recalibra gates ainda.
- Apenas transforma observabilidade em decisão operacional automatizada.
