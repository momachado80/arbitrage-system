# Shadow Exit Kill Window 180 — Deliverable

## A. Resumo do que foi implementado

Novo challenger **`shadow_1000_structural_exitkill_window180_v1`**, idêntico ao `shadow_1000_structural_exitkill_v1` exceto por **uma única alteração**: `monitoringWindowMs` de 90_000 para **180_000** (3 minutos).

**Implementado:**
1. **Profile** com mesma entrada estrutural (pair set, fill bucket, capfloor, degratio, sizing)
2. **Diagnostics** dedicados (`structuralExitKillWindow180Diagnostics.ts`) com earlyKillExitCount, killReasonCounts, avgHoldingTimeMs, avgHoldingMsEarlyKill
3. **Audit route** expondo `structuralExitKillWindow180Diagnostics`, `structuralExitKillWindow180Comparison`, `structuralExitKillWindow180CausalAudit`
4. **Comparisons** vs shadow_1000, shadow_1000_structural_riskmanaged_v1, shadow_1000_structural_exitkill_v1
5. **Pipeline** (snapshot, brief, judge) integrada ao novo challenger

---

## B. Arquivos alterados

| Arquivo | Alteração |
|---------|-----------|
| `dashboard/lib/shadowSimulationProfiles.ts` | Novo profile `shadow_1000_structural_exitkill_window180_v1` |
| `dashboard/lib/structuralExitKillWindow180Diagnostics.ts` | **Novo** — diagnostics, comparison, causal audit |
| `dashboard/app/api/shadow/audit/route.ts` | Exposição dos blocos window180 |
| `dashboard/scripts/shadow-snapshot.ts` | Captura dos blocos window180 |
| `dashboard/scripts/shadow-brief.ts` | Seção "Exit Kill Window 180" |
| `dashboard/scripts/shadow-judge.ts` | Judgment em `judgment_exitkill_window180_latest.json` |

---

## C. Hipótese causal exata

> **Se a causa principal do earlyKillExitCount=0 no exitkill_v1 foi a janela curta (90s), então** ampliar a janela para 180s **dará chance real à lógica de kill atuar**, permitindo testar se a saída adaptativa reduz destruição econômica quando aplicável.

- **Variável isolada:** `monitoringWindowMs` (90s → 180s)
- **Objetivo primário:** Verificar se `earlyKillExitCount > 0`
- **Objetivo secundário:** Comparar PnL vs exitkill_v1 e vs baseline

---

## D. Comandos para validar

**Localmente:**
```bash
cd dashboard
npm run build
npm run dev
# Após boot + alguns ciclos:
curl -s http://localhost:3000/api/shadow/audit | jq '.structuralExitKillWindow180Diagnostics'
curl -s http://localhost:3000/api/shadow/audit | jq '.structuralExitKillWindow180Comparison'
```

**Produção:**
```bash
# Snapshot (inclui window180)
npm run shadow:snapshot

# Brief (seção Exit Kill Window 180)
npm run shadow:brief

# Judge (saída judgment_exitkill_window180_latest.json)
npm run shadow:judge
```

**Checar profile no código:**
```bash
rg "shadow_1000_structural_exitkill_window180_v1" dashboard/
```

**Verificar se o kill disparou:**
```bash
curl -s <AUDIT_URL>/api/shadow/audit | jq '.structuralExitKillWindow180Diagnostics | {earlyKillExitCount, killReasonCounts, avgHoldingTimeMs, avgHoldingMsEarlyKill}'
```

---

## E. Risco de interpretação errada

1. **earlyKillExitCount > 0 não implica melhoria econômica:** O kill pode estar cortando trades que teriam revertido. A métrica principal continua sendo totalRealizedPnL.
2. **Comparação vs exitkill_v1:** Ambos usam a mesma base de oportunidades; diferenças vêm só da janela. Se window180 tiver mais kills e PnL pior, a hipótese de “mais kill = melhor” falha.
3. **closedInKillWindow vs closedOutsideKillWindow:** Com janela 180s, mais trades fecham dentro da janela. Um `closedInKillWindow` alto com `earlyKillExitCount` ainda 0 indica que os critérios de kill continuam não disparando (thresholds ou regime).
