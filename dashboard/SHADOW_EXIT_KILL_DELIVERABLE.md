# Shadow Exit Kill Challenger — Deliverable

## A. Resumo do que foi implementado

Novo challenger `shadow_1000_structural_exitkill_v1` focado exclusivamente em **saída adaptativa mais agressiva**. Mantém a mesma entrada estrutural do `shadow_1000_structural_riskmanaged_v1` (pair set, fill bucket 0.1–0.25, capfloor 0.045, degRatioMin 0.24, sizing adaptativo), mas substitui o early thesis failure por uma lógica de exit kill mais objetiva e instrumentada.

**Implementado:**
1. **Profile** com `exitKillTarget` (janela 90s, critérios de kill configuráveis)
2. **Lógica de exit kill** em `runCycle`: opportunity absent, capturable decay, observed decay, net edge floor — em ordem de avaliação
3. **Campos de audit** em `ShadowTrade` e `ClosedTradeAuditEntry`: `exitKillTriggered`, `exitKillReason`, `exitKillAtMsFromOpen`, `capturableEdgeAtKill`, `observedEdgeAtKill`, `degradationRatioAtKill`, `opportunityAbsentCyclesAtKill`
4. **Diagnostics** (`structuralExitKillDiagnostics.ts`): bloco completo + comparison vs shadow_1000, shadow_1000_adapt_captrade_exitrefine_v1, shadow_1000_structural_riskmanaged_v1
5. **Endpoint** `/api/shadow/audit`: `structuralExitKillDiagnostics` e `structuralExitKillComparison`
6. **Esteira de validação**: snapshot captura blocos exit kill; brief tem seção opcional; judge produz `judgment_exitkill_latest.json` quando dados existem

---

## B. Arquivos alterados

| Arquivo | Alteração |
|---------|-----------|
| `dashboard/lib/shadowSimulationProfiles.ts` | Profile `shadow_1000_structural_exitkill_v1` com `exitKillTarget` |
| `dashboard/lib/shadowSimulationService.ts` | Lógica exit kill em runCycle; `recordExitKillEvaluated` após gate estrutural |
| `dashboard/lib/shadowSimulationStore.ts` | Campos exit kill em `ShadowTrade` |
| `dashboard/lib/shadowClosedTradeAudit.ts` | Campos exit kill em `ClosedTradeAuditEntry` e `toAuditEntry` |
| `dashboard/lib/structuralExitKillDiagnostics.ts` | **Novo** — diagnostics e comparison |
| `dashboard/app/api/shadow/audit/route.ts` | Expor `structuralExitKillDiagnostics` e `structuralExitKillComparison` |
| `dashboard/scripts/shadow-snapshot.ts` | Capturar `structuralExitKillDiagnostics` e `structuralExitKillComparison` |
| `dashboard/scripts/shadow-brief.ts` | Seção opcional "Exit Kill Challenger" |
| `dashboard/scripts/shadow-judge.ts` | Judgment separado para exit kill em `judgment_exitkill_latest.json` |

---

## C. Hipótese causal exata do challenger novo

> **Se o problema remanescente está mais na saída do que na entrada, então** um challenger que preserve o subset estrutural principal, preserve o sizing adaptativo já conhecido e introduza uma lógica de kill/exit por deterioração mais efetiva **pode reduzir destruição econômica materialmente**.

- **Universe:** idêntico ao `shadow_1000_structural_riskmanaged_v1`
- **Mudança única:** saída adaptativa mais agressiva e objetiva (exit kill) em vez de early thesis failure
- **Variável dependente:** lucro líquido realizado (`totalRealizedPnL`, `avgRealizedPnL`)

---

## D. Critérios de kill escolhidos e por quê

| Critério | Valor | Motivo |
|----------|-------|--------|
| `killAbsentCycles` | 2 | Oportunidade ausente 2 ciclos consecutivos — tese desapareceu antes de 90s |
| `killCapturableDecayFraction` | 0.5 | Capturable proxy ≤ 50% do capturable na entrada — deterioração clara |
| `killObservedEdgeDecayFraction` | 0.5 | Observed ≤ 50% do observed na entrada — edge colapsou |
| `killNetEdgeFloor` | 0.02 | Net edge ≤ 2% — abaixo de threshold operacional mínimo |
| `monitoringWindowMs` | 90_000 | Janela 90s (60–120s sugerido) — foco em deterioração pós-entrada inicial |

**Ordem de avaliação:** (1) opportunity absent, (2) capturable decay, (3) observed decay, (4) net edge floor — primeiro que disparar fecha o trade com `exitReason: "exit_kill"` e reason específica.

---

## E. Como validar localmente e em produção

**Localmente:**
```bash
cd dashboard
npm run build
npm run dev
# Aguardar boot + ciclos com oportunidades
# Depois: curl http://localhost:3000/api/shadow/audit | jq '.structuralExitKillDiagnostics, .structuralExitKillComparison'
```

**Produção:**
```bash
# Snapshot
npm run shadow:snapshot

# Brief (inclui seção Exit Kill se dados existirem)
npm run shadow:brief

# Judge (saída structural_riskmanaged + exitkill)
npm run shadow:judge
```

---

## F. Comandos exatos

**Checar se o profile existe no código:**
```bash
rg "shadow_1000_structural_exitkill_v1" dashboard/
```

**Checar se foi exposto no endpoint:**
```bash
curl -s <AUDIT_URL>/api/shadow/audit | jq 'keys | map(select(. | test("exit|Exit")))'
# Deve incluir: structuralExitKillDiagnostics, structuralExitKillComparison
```

**Comparar com baseline:**
```bash
curl -s <AUDIT_URL>/api/shadow/audit | jq '.structuralExitKillComparison["shadow_1000"]'
curl -s <AUDIT_URL>/api/shadow/audit | jq '.structuralExitKillComparison["shadow_1000_structural_riskmanaged_v1"]'
```

---

## G. Riscos de interpretação errada dos dados

1. **earlyKillExitCount = 0 não implica falha:** Se a oportunidade raramente deteriora nos primeiros 90s, o exit kill pode não disparar — isso pode ser regime favorável, não bug.
2. **Confundir exit kill com early thesis:** Exit kill usa `exitReason: "exit_kill"` e campos `exitKill*`; early thesis é apenas para profiles sem `exitKillTarget`.
3. **Baseline fill ratio:** Baseline pode não ter `fillRatio` explícito; usamos fallback `filledCapital/requestedCapital`. Se `requestedCapital` for 0 ou ausente, fill ratio = 0.
4. **structuralRiskFilterMatchAtOpen:** Diagnostics filtra por `structuralRiskFilterMatchAtOpen !== false`. Trades do exit kill passam pelo gate estrutural, então devem ter esse campo true.
