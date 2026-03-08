# Capital Sizing & Executable Opportunity Quality — Analysis

> Diagnostic-only document. No production logic changes.

## TASK 1 & 2: Exact Code Path and Formula for recommendedCapital

### Capacity Engine (capitalCapacityEngine.ts)

```
grossEdge     = max(0, opportunity.edge)
spread        = max(0.01, opportunity.spread)
liquidity     = max(1, opportunity.liquidity)
minNet        = 0.005
fee           = 0.002

// solveMaxCapital:
edgeAfterFee  = grossEdge - fee
maxNet        = edgeAfterFee - minNet
slipCoeff     = (spread × 1.2) / liquidity
impactCoeff   = (1.5 × 0.8) / liquidity = 1.2 / liquidity
totalCoeff    = slipCoeff + impactCoeff = 1.2 × (spread + 1) / liquidity
maxOrderSize  = maxNet / totalCoeff = (grossEdge - 0.007) × liquidity / (1.2 × (spread + 1))
capped        = min(maxOrderSize, liquidity × 0.15, 5000)
maxCapital    = max(0, capped)

recommendedCapital = (netEdge > 0.005 && maxCapital > 0)
  ? min(maxCapital × 0.5, maxCapital)
  : 0
```

### Execution Engine (realisticExecutionEngine.ts)

```
recommendedRaw = min(
  capacity.recommendedCapital,
  profileState.availableCapital,
  profileState.maxCapitalPerTrade,
  liquidity × 0.1,                    // liquidity = max(1, opp.liquidity) × liquidityHaircut
  opportunity.liquidity × 0.08        // HARD CAP: 8% of raw liquidity
)

requestedCapital = min(recommendedRaw, remainingCluster, remainingMarket)
```

**Inputs affecting recommendedCapital:**
- edge, spread, liquidity (opportunity)
- confidence (affects capacityConfidence, not recommendedCapital directly)
- portfolio: availableCapital, maxCapitalPerTrade, exposure limits
- liquidityHaircut (0.6–0.65)

---

## TASK 3: Why recommendedCapital Is Small in Observed Examples

**Example A:** observedEdge ≈ 0.325, netEdgeAfterImpact ≈ 0.325, recommendedCapital ≈ 0.039

- Edge is high; the capacity engine would allow larger size if liquidity were sufficient.
- The binding constraint is almost certainly `opportunity.liquidity × 0.08 ≈ 0.039` → liquidity ≈ 0.49.
- Low liquidity (likely from thin graph markets or a low minLiquidity) drives the 8% cap down to ~\$0.04.

**Example B:** observedEdge ≈ 0.27, netEdgeAfterImpact ≈ 0.27, recommendedCapital ≈ 0.66

- `opportunity.liquidity × 0.08 ≈ 0.66` → liquidity ≈ 8.25.
- Slightly higher liquidity, but still small; the 8% rule again dominates.

**Conclusion:** The execution engine’s caps `liquidity × 0.1` and `opportunity.liquidity × 0.08` are the main limiters when liquidity is low. Strong edge does not compensate for thin books.

---

## TASK 4: Main Bottleneck

**D. Combination**

- **A. Capital sizing is restrictive:** Yes. The 8% and 10% liquidity caps in the execution engine are strict relative to available book depth.
- **B. Scanner surfaces non-executable opportunities:** Partially. Graph opportunities use `minLiquidity` across legs; some high-edge clusters have one very thin market.
- **C. Execution model filters correctly:** Yes. It correctly blocks trades that would be tiny or illiquid in practice.

---

## TASK 5: Scanner vs Execution Alignment

**Misalignment:** Yes.

- **rankOpportunities:** `compositeScore = edge × liquidityScore(liquidity) × confidence`
  - liquidityScore = min(1, log10(liquidity)/6) — grows slowly with liquidity.
- **rankGraphOpportunities:** `compositeScore = edge × log10(1 + liquidity) × confidence`
  - Similar: edge-heavy, liquidity has limited weight.

- **Execution:** Uses `opportunity.liquidity × 0.08` as a hard cap.

- **Effect:** High-edge, low-liquidity opportunities rank well but produce negligible executable size. Ranking favors edge; execution depends heavily on liquidity.

---

## TASK 6: Proposed Future Metric

**executableExpectedValue (EEV)**

```
EEV = recommendedCapital × fillProbability × netEdgeAfterImpact
```

Why:
- Combines size (recommendedCapital), fill realism (fillProbability), and edge (netEdgeAfterImpact).
- Directly measures expected dollar profit per opportunity.
- Aligns ranking with economic impact instead of raw edge.
- Alternatives: `economicExecutabilityScore` (EEV normalized) or `expectedProfitAfterFill` (same as EEV).

**Usage:** Use EEV (or a normalized version) in ranking so opportunities with high edge but low executable value are down-ranked.
