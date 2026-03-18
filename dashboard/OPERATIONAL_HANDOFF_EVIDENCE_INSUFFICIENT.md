# Handoff Operacional — Evidência Insuficiente (Branch A)

**Data da leitura:** 2025-03  
**Fonte:** Produção `https://dashboard-next-production-b126.up.railway.app`  
**Ramo da árvore:** A — `evidenceSufficient = false`

---

## 1. Leitura objetiva do porquê a evidência ainda é insuficiente

O `structuralRecalibrationReview` em produção retornou:

```
recommendedNextSingleHypothesis.reason: "Amostra insuficiente: 7 profiles, 350 raw opps, 2 ciclos. Mínimos: 2 profiles, 50 raw opps, 10 ciclos."
evidenceSufficient: false
executiveSummary: "Sem evidência suficiente ainda para priorizar recalibração. Continuar coleta de dados."
```

**Causa:** O critério de **ciclos** não foi atingido. O sistema requer no mínimo **10 ciclos processados** para considerar a amostra suficiente. A produção está em **2 ciclos**.

Os demais critérios foram satisfeitos:
- **Profiles estruturais:** 7 (mínimo: 2) ✓
- **Opps brutas agregadas:** 350 (mínimo: 50) ✓
- **Ciclos:** 2 (mínimo: 10) ✗

Com apenas 2 ciclos, todos os profiles recebem `dominantChokeStage: "insufficient_sample"` e `dominantChokeReason: "poucos ciclos"`. O judge bloqueia conclusões sobre choke dominante porque a janela temporal é pequena demais.

---

## 2. Quanto falta para atingir suficiência

| Critério      | Atual | Mínimo | Falta                      |
|---------------|-------|--------|----------------------------|
| Profiles      | 7     | 2      | 0 (já suficiente)          |
| Raw opps      | 350   | 50     | 0 (já suficiente)          |
| **Ciclos**    | **2** | **10** | **8 ciclos**               |

**Tempo estimado para suficiência:** Com ciclo de 10s, 8 ciclos ≈ **80 segundos** de execução contínua. Em prática, o shadow loop precisa rodar sem reinício por ~2 minutos para acumular 10 ciclos.

**Nota:** Os `closedTradeCount` altos (500, 327, 71, etc.) indicam reidratação de histórico persistido — não são closes do fluxo atual. O `cyclesProcessed: 2` reflete o estado real do runtime desde o último boot.

---

## 3. Comandos de validação para rechecagem posterior

```bash
# URL de produção
DASHBOARD_URL="https://dashboard-next-production-b126.up.railway.app"

# Review completo
curl -s "${DASHBOARD_URL}/api/shadow/audit" | jq '.structuralRecalibrationReview'

# Apenas evidência e hipótese
curl -s "${DASHBOARD_URL}/api/shadow/audit" | jq '.structuralRecalibrationReview | {evidenceSufficient, recommendedNextSingleHypothesis, executiveSummary}'

# Ciclos por profile estrutural (max = ciclos desde boot)
curl -s "${DASHBOARD_URL}/api/shadow/audit" | jq '[.profileEligibilityDiagnostics["shadow_1000_structural_riskmanaged_v1"].cyclesProcessed]'

# Verificar quando evidenceSufficient vira true (ciclos >= 10)
curl -s "${DASHBOARD_URL}/api/shadow/audit" | jq '.structuralRecalibrationReview.evidenceSufficient'
```

**Rechecagem recomendada:** Após 2–3 minutos de execução contínua do shadow loop, rodar novamente. Se `evidenceSufficient` passar a `true`, a árvore de decisão poderá seguir para o ramo B ou C na próxima análise.

---

## 4. Conclusão explícita

**`do_not_modify_yet`**

- Não criar challenger novo.
- Não recalibrar gates.
- Manter coleta de dados.
- Rechecar após acumular ≥10 ciclos.

---

## Prévia dos dados quando houver suficiência

Os diagnósticos já mostram um padrão consistente nos profiles estruturais risk-managed:

- `shadow_1000_structural_riskmanaged_v1`: pairEligibleCount=1, fillEligibleCount=0, choke "choke em fill bucket"
- Demais estruturais (exitkill, lateexit): mesmo padrão — passam pair (1 opp) e morrem em fill

O descarte dominante antes do fill é `fill_rejected` (89) vindo do `simulateRealisticEntry`. O `structural_risk_fill_bucket_mismatch` (3) é o gate específico estrutural. Quando houver evidência suficiente, o choke em **fill_gate** tende a ser priorizado. Por ora, porém, a decisão permanece: **do_not_modify_yet**.
