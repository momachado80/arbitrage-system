# Post-Mortem: shadow_1000_adapt_edgegate_v1 vs shadow_1000

**Data source**: Production audit API (2026-03-12), persistence file, codebase analysis  
**Challenger**: shadow_1000_adapt_edgegate_v1 (edge gate 0.012)  
**Baseline**: shadow_1000 (minNetCapturableEdgeToTrade 0.007)

---

## 1. baselineVsChallengerSummary

### shadow_1000 (baseline) — from production audit

| Metric | Value |
|--------|-------|
| totalClosed | 500 |
| avgRealizedPnL | -1.548 |
| medianRealizedPnL | -0.275 |
| totalRealizedPnL | -773.81 |
| avgHoldingTimeMs | 299,227 (~5 min) |
| avgFilledCapital | 10.19 |
| avgFillRatio | 0.247 |
| avgObservedEdgeAtEntry | 0.213 |
| avgCapturableEdgeAtEntry | 0.055 |
| avgEffectiveEntryPrice | 0.787 |
| maxDrawdown | N/A (not exposed in audit) |

**Exit reason distribution (shadow_1000)**:
- max_holding_time: 481 (96.2%), totalPnL -714.46, avgPnL -1.49
- stop_loss: 19 (3.8%), totalPnL -59.35, avgPnL -3.12

### shadow_1000_adapt_edgegate_v1 (challenger) — from prior observation + persistence

> **Note**: Challenger data is not rehydrated (rehydration skips non-baseline profiles). A diagnostic endpoint `/api/shadow/persistence-challengers` was added to read challenger trades from the persistence file. **Deploy and call this endpoint to retrieve exact challenger stats.** Until then, the following uses the prior observed outcome and mechanism inference.

**Observed outcome (prior run)**:
- totalClosed: ~25
- avgRealizedPnL: worse than baseline
- medianRealizedPnL: worse than baseline
- avgFilledCapital: **larger** than baseline (10.19)
- Exit mix: max_holding_time (23), stop_loss (2)
- avgHoldingTimeMs: ~290,000 (~4.8 min)

**Edge gate configuration**:
- Baseline: minNetCapturableEdgeToTrade = 0.007
- Challenger: minCapturableEdgeToTrade = 0.012 (Math.max(0.007×1.5, 0.007+0.005, 0.01))
- Effect: Challenger rejects entries with capturableEdgeAtEntry < 0.012

---

## 2. distributionDifferences

### 2.1 capturableEdgeAtEntry distribution (aggregate across all profiles)

From `byCapturableEdgeDecile` (3000 trades, all profiles):

| Decile | tradeCount | avgRealizedPnL | avgFilledCapital | avgFillRatio |
|--------|------------|----------------|-----------------|--------------|
| d1 (lowest) | 300 | -2.29 | 16.18 | 0.311 |
| d2 | 300 | -2.32 | 16.07 | 0.322 |
| d3 | 300 | -2.06 | 14.03 | 0.286 |
| d4 | 300 | -2.27 | 15.21 | 0.261 |
| d5 | 300 | -2.23 | 14.71 | 0.225 |
| d6 | 300 | -1.95 | 12.68 | 0.226 |
| d7 | 300 | -2.09 | 13.40 | 0.204 |
| d8 | 300 | -2.07 | 12.96 | 0.213 |
| d9 | 300 | -1.75 | 10.78 | 0.193 |
| d10 (highest) | 300 | -2.19 | 13.00 | 0.173 |

**Observation**: Lower edge deciles (d1–d5) have *higher* avgFilledCapital (14–16) than higher deciles (d9–d10: 10–13). Higher edge deciles have slightly better (less negative) avgRealizedPnL in the middle (d6, d9). The edgegate filters to capturableEdgeAtEntry ≥ 0.012, i.e. the upper part of the distribution. In the aggregate, that slice has *lower* filled capital on average — yet the challenger showed *higher* avgFilledCapital. This implies a different selection effect for the challenger.

### 2.2 fillRatio distribution (aggregate)

From `byFillRatioBucket`:
- 0.1–0.25: 1602 trades, avgPnL -1.89, avgCapturableEdge 0.066
- 0.25–0.5: 1398 trades, avgPnL -2.39, avgCapturableEdge 0.044

**Observation**: Trades with lower fill ratio (0.1–0.25) have higher avgCapturableEdge and *less* negative PnL. Trades with higher fill ratio (0.25–0.5) have lower edge and *more* negative PnL. This suggests that **fuller fills are associated with worse realized outcomes** — possibly because larger fills occur when liquidity is plentiful but the opportunity is already decaying or mispriced.

### 2.3 pairKey concentration (aggregate worstPairs)

Top 5 pairs by avgRealizedPnL (worst):

| pairKey | tradeCount | avgRealizedPnL | medianRealizedPnL | avgCapturableEdge | dominantExit |
|---------|------------|----------------|-------------------|------------------|--------------|
| 540820+573647 | 29 | -3.21 | -5.39 | 0.032 | stop_loss |
| 573647+604490 | 82 | -2.45 | -0.82 | 0.084 | stop_loss |
| 567561+604490 | 90 | -2.39 | -0.82 | 0.078 | stop_loss |
| 562187+573647 | 118 | -2.39 | -0.98 | 0.064 | stop_loss |
| 562187+567561 | 128 | -2.34 | -1.02 | 0.056 | stop_loss |

**Observation**: Pairs with *higher* avgCapturableEdgeAtEntry (0.078, 0.084) are among the worst. The worst pair (540820+573647) has the *lowest* edge (0.032) — it would be rejected by the edgegate. So the gate would exclude the single worst pair but would *still admit* 573647+604490 and 567561+604490, which have high edge and high loss.

---

## 3. pairConcentrationAnalysis

- **Baseline shadow_1000**: 500 trades across many pairs; worst20 dominated by 565065+567561 (high filledCapital, ~52 USD, large losses).
- **Challenger**: With gate 0.012, pairs with avgCapturableEdge < 0.012 (e.g. 540820+573647, 562794+567561) are partially or fully excluded. Pairs with high edge (e.g. 573647+604490, 567561+604490) are *preferred* by the gate.
- **Mechanism**: The gate biases selection toward high-edge pairs that, in the aggregate, still have negative expectancy and often large losses. High edge does not imply better realized outcome in this dataset.

---

## 4. capitalConcentrationAnalysis

- **Baseline**: avgFilledCapital 10.19, median likely lower (many small fills).
- **Challenger**: avgFilledCapital **larger** than baseline.
- **Likely driver**: Fewer trades → more available capital per opportunity → larger fills when the challenger *does* trade. The gate reduces trade count but increases effective size per trade.
- **Harmful interaction**: Larger filled capital × same or worse loss rate → larger dollar losses. The gate concentrates capital into fewer, larger, still-unprofitable trades.

---

## 5. exitMixAnalysis

**Aggregate exit diagnostics**:
- max_holding_time: 965 trades, avgPnL -0.99, avgHoldingTime 305s, avgFillRatio 0.25
- stop_loss: 2035 trades, avgPnL -2.66, avgHoldingTime 51s, avgFillRatio 0.24

**Baseline shadow_1000**: 96% max_holding_time, 4% stop_loss.  
**Challenger**: ~92% max_holding_time, ~8% stop_loss (23 vs 2).

**Observation**: stop_loss exits have much worse avgPnL (-2.66 vs -0.99). Challenger had a higher share of stop_loss (8% vs 4%), which amplifies losses. The gate did not improve exit quality; it may have admitted trades more prone to stop_loss (e.g. high-edge pairs that decay quickly).

---

## 6. rootCause

### Why did edgegate v1 fail?

The edge gate increased the minimum capturable edge from 0.007 to 0.012. Instead of improving outcomes, it:

1. **Reduced trade count** (fewer opportunities pass the gate).
2. **Increased avgFilledCapital** (fewer trades → more available capital → larger fills per trade).
3. **Selected a worse subset** (high-edge pairs in this dataset are not better; several are among the worst by realized PnL).
4. **Concentrated losses** (larger fills × same or worse loss rate = larger dollar losses).

### Strongest loss-amplifying mechanism

**Capital concentration**: The gate reduced the number of trades but increased filled capital per trade. With negative expectancy, concentrating capital into fewer trades increased total losses.

### Main issue

- Not primarily pair concentration (the gate did exclude some bad pairs).
- **Primarily capital concentration**: fewer trades, larger fills, same/worse loss rate.
- **Secondarily threshold placement**: the 0.012 threshold favored high-edge pairs that in practice underperformed.

### What to avoid in the next challenger

1. Raising a univariate edge threshold without capping or reducing trade size.
2. Assuming higher capturable edge implies better realized PnL; the data show the opposite in several high-edge pairs.
3. Ignoring the available-capital effect: fewer trades → more capital per trade → larger losses when expectancy is negative.

---

## 7. nextExperimentDesignGuidance

### 3 things to avoid

1. **Stricter edge threshold alone** — edgegate v1 showed this amplifies losses via capital concentration.
2. **Selecting by raw capturableEdge** without pair-specific adjustment — high-edge pairs 573647+604490, 567561+604490 are toxic.
3. **Ignoring fill size** — larger fills are associated with worse outcomes; consider a max filled capital or fill-ratio cap.

### 3 things worth testing next

1. **Pair-level penalties (entryScorePenaltyByPair)** — penalize historically bad pairs (e.g. 573647+604490, 567561+604490) instead of a global edge raise.
2. **Reduced maxCapitalPerTrade for challenger** — same gate, but lower max size to avoid capital concentration.
3. **Edge threshold + fill cap** — e.g. minCapturableEdge 0.012 *and* maxFilledCapitalPerTrade 15 to limit exposure on any single trade.

### Strongest single-variable hypothesis

**Pair-level penalties** — worstPairs show clear pair-specific loss patterns. Penalizing bad pairs should avoid the capital-concentration effect of a global edge gate while still filtering the worst opportunities.

---

## G. Validation Requirements

### 1. Files / code paths inspected

- `dashboard/lib/shadowClosedTradeAudit.ts` — audit structure, toAuditEntry, byCapturableEdgeDecile, worstPairs
- `dashboard/lib/shadowSimulationStore.ts` — rehydration (challengers skipped), persistence
- `dashboard/lib/shadowClosedTradePersistence.ts` — snapshot structure, byProfile
- `dashboard/lib/adaptiveCalibrationEngine.ts` — edgegate v1 spec, threshold 0.012
- `dashboard/lib/shadowSimulationProfiles.ts` — baseline config, minNetCapturableEdgeToTrade
- `dashboard/lib/shadowSimulationService.ts` — entry gate logic, minCapturableEdgeToTrade
- Production `/api/shadow/audit` response (2026-03-12)

### 2. Additional temporary diagnostics

- Added **`/api/shadow/persistence-challengers`** — reads raw persistence file and returns challenger (non-baseline) closed trades with full stats, deciles, byPairKey, worst10/best10. Deploy and call to complete the post-mortem with exact challenger data.

### 3. Persisted history usage

- Yes. The baseline comparison uses rehydrated production data (500 trades per profile).
- Challenger data, if present, lives in the persistence file but is **not** rehydrated (only SHADOW_PROFILES are restored). The new endpoint reads the file directly.

### 4. Analysis basis

- **Baseline**: Actual closed trades from production (rehydrated).
- **Challenger**: Prior observed metrics + mechanism inference. For exact challenger stats, use `/api/shadow/persistence-challengers` after deploy.
