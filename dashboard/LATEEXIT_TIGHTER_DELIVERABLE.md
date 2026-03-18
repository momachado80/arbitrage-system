# Challenger: shadow_1000_structural_lateexit_tighter_v1

## A. Hipótese causal exata

**Hipótese:** Com thresholds de late exit apertados (stagnantEdgeFloor 3%→4.5%, netEdgeProlongedFloor 2%→3%), a saída tardia passará a disparar com mais frequência na janela 90–300s, reduzindo o share de closes por max_holding_time e potencialmente reduzindo perdas.

**Evidência base:** A auditoria causal do `shadow_1000_structural_lateexit_nonreversion_v1` mostrou que o edge observado no close permanece alto (~25%), acima dos thresholds atuais (3%, 2%), então nenhum late exit foi acionado. Apertar os thresholds alinha a regra ao regime observado.

## B. Arquivos alterados

| Arquivo | Alteração |
|---------|-----------|
| `lib/shadowSimulationProfiles.ts` | Novo profile `shadow_1000_structural_lateexit_tighter_v1` |
| `lib/structuralLateExitTighterDiagnostics.ts` | **NOVO** – diagnostics e comparison |
| `lib/structuralLateExitCausalAudit.ts` | Suporte a opts (profileId, thresholds) |
| `app/api/shadow/audit/route.ts` | Exposição de tighter diagnostics/comparison/causal |
| `scripts/shadow-snapshot.ts` | Blocos late exit tighter |
| `scripts/shadow-brief.ts` | Seção late exit tighter |
| `scripts/shadow-judge.ts` | Judgment late exit tighter |

## C. Mudança única

- `stagnantEdgeFloor`: 0.03 → 0.045
- `netEdgeProlongedFloor`: 0.02 → 0.03

Demais parâmetros idênticos ao `shadow_1000_structural_lateexit_nonreversion_v1`.

## D. Comandos para verificar

```bash
# API audit (tighter)
curl -s "https://dashboard-next-production-b126.up.railway.app/api/shadow/audit" | jq '.structuralLateExitTighterDiagnostics'

# Comparação vs non-reversion
curl -s "https://dashboard-next-production-b126.up.railway.app/api/shadow/audit" | jq '.structuralLateExitTighterComparison["shadow_1000_structural_lateexit_nonreversion_v1"]'

# Causal audit
curl -s "https://dashboard-next-production-b126.up.railway.app/api/shadow/audit" | jq '.structuralLateExitTighterCausalAudit.conclusion'
```

## E. Risco de interpretação errada

1. **Thresholds excessivamente apertados:** Com 4.5% e 3%, o late exit pode disparar muito cedo, antes de qualquer “reversão útil”, e sair de trades que ainda evoluiriam bem.
2. **Regime diferente:** Se o regime mudar e o edge cair mais que antes, os novos thresholds podem acionar com frequência indesejada.
3. **Confusão de baseline:** O judge compara primeiro vs `shadow_1000_structural_lateexit_nonreversion_v1` (baseline direta). Interpretar vs `shadow_1000` sem levar em conta o non-reversion pode levar a conclusões incorretas.
4. **Tamanho de amostra:** Com poucos closes, diferenças entre tighter e non-reversion podem ser ruído.
