# Shadow Structural Risk-Managed Challenger — Deliverable

## Patch Summary

Novo challenger **shadow_1000_structural_riskmanaged_v1** adicionado ao shadow simulation, sem alterar profiles ou lógica existente:

- **Gate estrutural**: pairKeys permitidos (7 pares) + fillRatioBucket `0.1-0.25`
- **Entrada**: capturableEdge ≥ 4.5%, entryDegradationRatio ≥ 0.24
- **Sizing adaptativo**: multiplier 0.25–1.0 (cap < 5.5% → ×0.6, deg < 0.26 → ×0.7, acumulado)
- **Early thesis-failure exit**: janela 90s; triggers: edge decay, net edge abaixo de metade, opp ausente 2 ciclos
- **Diagnósticos**: `structuralRiskManagedDiagnostics` e `structuralRiskManagedComparison` vs baseline e exitrefine

---

## Novos campos em ShadowTrade / ClosedTradeAuditEntry

```ts
// Persistidos na abertura
structuralRiskCapitalMultiplierAtOpen?: number | null
structuralRiskCapfloorAtOpen?: number | null
structuralRiskDegRatioAtOpen?: number | null
structuralRiskTargetPairSetAtOpen?: string[] | null
structuralRiskTargetFillBucketAtOpen?: string | null
structuralRiskFilterMatchAtOpen?: boolean
structuralRiskTargetVersion?: string | null

// Persistidos no fechamento (early thesis failure)
earlyThesisFailureTriggered?: boolean
earlyThesisFailureReason?: string | null
earlyThesisFailureAtMsFromOpen?: number | null
```

---

## Blocos de audit adicionados

### structuralRiskManagedDiagnostics

```ts
{
  profileId: string
  evaluatedOpportunityCount: number
  openedTradeCount: number
  rejectedByStructuralPairCount: number
  rejectedByFillBucketCount: number
  rejectedByCapfloorCount: number
  rejectedByDegRatioCount: number
  rejectedByOtherReasonCount: number
  avgCapturableEdgeOpened: number
  avgObservedEdgeOpened: number
  avgDegradationRatioOpened: number
  avgCapitalMultiplierOpened: number
  minCapitalMultiplierOpened: number
  p10CapitalMultiplierOpened: number
  p50CapitalMultiplierOpened: number
  p90CapitalMultiplierOpened: number
  pmaxCapitalMultiplierOpened: number
  earlyThesisFailureExitCount: number
  avgHoldingMsEarlyThesisFailure: number
  avgRealizedPnL: number
  medianRealizedPnL: number
  totalRealizedPnL: number
  avgFillRatioOpened: number
  avgRealizedPnL_earlyExit: number
  countEarlyExit: number
}
```

### structuralRiskManagedComparison

`Record<string, StructuralRiskManagedComparisonBlock>` para:
- `shadow_1000`
- `shadow_1000_adapt_captrade_exitrefine_v1`

Cada bloco contém: `baselineProfileId`, `challengerProfileId`, `baselineClosed`, `challengerClosed`, `baselineAvgRealizedPnL`, `challengerAvgRealizedPnL`, `baselineMedianRealizedPnL`, `challengerMedianRealizedPnL`, `baselineTotalRealizedPnL`, `challengerTotalRealizedPnL`, `baselineAvgFillRatio`, `challengerAvgFillRatio`, `baselineAvgCapturableEdgeAtEntry`, `challengerAvgCapturableEdgeAtEntry`, `sameUniverseNote`.

---

## Novos rejection reasons em rejectionCountsByProfile

- `structural_risk_pair_mismatch`
- `structural_risk_fill_bucket_mismatch`
- `structural_risk_capfloor`
- `structural_risk_degratio`

---

## Comando curl + jq para inspeção

```bash
# Local (ajuste AUDIT_URL conforme seu ambiente)
AUDIT_URL="http://localhost:3000/api/shadow/audit"

curl -sS "$AUDIT_URL" | jq '{
  operationalTruth,
  profileSummaries: [.profileSummaries[]? | select(.profileId == "shadow_1000" or .profileId == "shadow_1000_adapt_captrade_exitrefine_v1" or .profileId == "shadow_1000_structural_riskmanaged_v1")],
  structuralRiskManagedDiagnostics,
  structuralRiskManagedComparison,
  rejectionCountsByProfile
}'
```

Para produção (Railway/Vercel):

```bash
AUDIT_URL="https://your-app.railway.app/api/shadow/audit"
curl -sS "$AUDIT_URL" | jq '{
  operationalTruth,
  profileSummaries: [.profileSummaries[]? | select(.profileId == "shadow_1000" or .profileId == "shadow_1000_adapt_captrade_exitrefine_v1" or .profileId == "shadow_1000_structural_riskmanaged_v1")],
  structuralRiskManagedDiagnostics,
  structuralRiskManagedComparison,
  rejectionCountsByProfile
}'
```

---

## Deploy

- **Railway conectado à main**: `git push origin main` dispara o deploy automaticamente.
- **Deploy manual**: rodar `npm run build` no `dashboard/` e fazer deploy da pasta `.next` no seu provedor; as variáveis de ambiente e comandos de start devem seguir a configuração existente.
