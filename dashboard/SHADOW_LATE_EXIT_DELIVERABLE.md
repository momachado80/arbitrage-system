# Shadow Late Exit Non-Reversion — Deliverable

## A. Hipótese causal exata

> **Se os trades não colapsam cedo, mas também não revertem de forma útil**, então uma saída por não reversão ou estagnação tardia **pode reduzir destruição** melhor do que uma lógica de kill precoce (que nunca atua porque trades fecham fora da janela).

- **Variável isolada:** critérios de saída tardia (após 90s)
- **Universe:** idêntico ao structural risk managed
- **Critérios objetivos:** non_reversion, stagnant_edge, net_edge_prolonged_low, opportunity_absent_late

## B. Arquivos alterados

| Arquivo | Alteração |
|---------|-----------|
| `lib/shadowSimulationProfiles.ts` | lateExitTarget interface + profile |
| `lib/shadowSimulationStore.ts` | lateExitTriggered, lateExitReason, etc. |
| `lib/shadowClosedTradeAudit.ts` | late exit audit fields |
| `lib/shadowSimulationService.ts` | late exit logic em runCycle |
| `lib/structuralLateExitDiagnostics.ts` | **Novo** |
| `app/api/shadow/audit/route.ts` | structuralLateExitDiagnostics, structuralLateExitComparison |
| `scripts/shadow-snapshot.ts` | late exit blocks |
| `scripts/shadow-brief.ts` | seção Late Exit |
| `scripts/shadow-judge.ts` | judgment_lateexit_latest.json |

## C. Por que essa hipótese é melhor alinhada ao regime

A auditoria mostrou que:
- Trades fecham **sempre** fora da janela de kill (90s/180s)
- avg holding ~300s (max_holding_time)
- Kill precoce **nunca** é avaliado

A hipótese de late exit opera **exatamente na fase em que os trades vivem**: após 90s. Em vez de tentar matar cedo (que não ocorre), ela identifica:
1. **non_reversion:** observed < 50% do entry após 90s
2. **stagnant_edge:** edge <= 3% por 3 ciclos consecutivos
3. **net_edge_prolonged_low:** edge <= 2%
4. **opportunity_absent_late:** opp ausente 2 ciclos na fase tardia

## D. Comandos para validar

```bash
# Local
cd dashboard && npm run dev
curl -s http://localhost:3000/api/shadow/audit | jq '.structuralLateExitDiagnostics'

# Produção
curl -s "https://dashboard-next-production-b126.up.railway.app/api/shadow/audit" | jq '.structuralLateExitDiagnostics'
curl -s "https://dashboard-next-production-b126.up.railway.app/api/shadow/audit" | jq '.structuralLateExitComparison'
```
