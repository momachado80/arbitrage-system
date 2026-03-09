# Available Capital Investigation — Diagnostic Summary

> Diagnostic-only document. No production logic changes.

## TASK 1: Code Path for availableCapital

**Source:** `dashboard/lib/shadowSimulationStore.ts`

- **Formula (lines 148, 186, 210):**  
  `availableCapital = Math.max(0, currentEquity - reservedCapital)`

- **Updated when:**
  1. `addShadowTrade` (line 148): after adding a new active trade
  2. `closeShadowTrade` (line 186): after closing a trade and updating realized PnL
  3. `updateShadowUnrealized` (line 210): when unrealized PnL is recalculated

- **Used by:** `shadowSimulationService.ts` → passes `profileState.availableCapital` to `realisticExecutionEngine.simulateRealisticEntry`

- **Caps in engine** (realisticExecutionEngine.ts line 139):  
  `recommendedRaw = Math.min(capCapacity, capAvailable, capMaxTrade, capLiq10Pct, capLiq8Pct)`

---

## TASK 2 & 3: What availableCapital Means and Formula

**Meaning:** Free capital available for new trades: equity minus capital already reserved by active positions.

**Formula (pseudo-code):**
```
currentEquity   = startingCapital + realizedPnL + unrealizedPnL
reservedCapital = sum(trade.filledCapital for each active trade)
availableCapital = max(0, currentEquity - reservedCapital)
```

**Dependencies:**
- `startingCapital`: from profile config (100 or 1000 USD)
- `realizedPnL`: sum of PnL from closed trades
- `unrealizedPnL`: mark-to-market of active trades
- `reservedCapital`: sum of `filledCapital` of all active trades

**No direct use of:** exposure limits (those cap `requestedCapital`, not `availableCapital`).

---

## TASK 4: Why availableCapital Becomes So Small

Values like `9.9e-10`, `3.8e-7`, `8.99e-8` occur when:

1. **Capital fully allocated:** `reservedCapital ≈ currentEquity`  
   - Many active shadow trades consuming almost all equity  
   - `availableCapital = currentEquity - reservedCapital` → tiny remainder from floating-point arithmetic

2. **Example:**  
   - `currentEquity = 100`  
   - `reservedCapital = 99.999999999` (many small `+= filledCapital` steps)  
   - `availableCapital = 100 - 99.999999999 ≈ 9.9e-10`

3. **Effect:** `capAvailable` becomes the binding cap in `Math.min(...)`, so `requestedCapital` is near zero, fill is rejected, and opportunities cannot open new trades.

---

## TASK 5: Interpretation

**B. Capital almost fully consumed by shadow positions** (plus floating-point dust)

- `availableCapital` behaves as designed.  
- Profiles have limited capital (100 or 1000 USD) and many small shadow trades, so most equity is reserved.  
- The very small values are floating-point leftovers when `reservedCapital` is almost equal to `currentEquity`.

Not explained by: unit mismatch (values are in USD), nor an obvious calculation bug.

---

## TASK 6: State Values When availableCapital Is Near Zero

| Variable          | Typical when low | Source                    |
|-------------------|------------------|---------------------------|
| portfolioEquity    | ~100 or ~1000    | startingCapital + PnL      |
| reservedCapital   | ≈ currentEquity  | sum(active filledCapital) |
| availableCapital  | &lt; 1, often &lt; 1e-6 | equity - reserved          |
| activeTrades      | many             | count of open trades      |
| freeCapitalRatio | &lt; 1%           | available / starting       |

**Caps applied in engine:**
- `capCapacity` (capacity engine)
- `capAvailable` = availableCapital  ← often the binder
- `capMaxTrade`, `capLiq10Pct`, `capLiq8Pct`

---

## TASK 7: Diagnostic Logs

- **Prefix:** `[DIAGNOSTICS] AVAILABLE CAPITAL ANALYSIS`
- **Where:** `shadowSimulationService.ts` (when `availableCapital < 5` before evaluate) and `botRunner.ts` snapshot (when any profile has `availableCapital < 5`).
- **Fields:** `portfolioEquity`, `reservedCapital`, `availableCapital`, `activeTrades`, `startingCapital`, `freeCapitalRatio`, `bindingCap`.

---

## TASK 8: Next Steps Toward a Profitable System

1. **Observe** the new capital diagnostics in production.
2. **Confirm** that `reservedCapital ≈ currentEquity` when available is tiny.
3. **Decide** whether to:
   - increase `startingCapital` for shadow profiles,
   - add a small floor for `availableCapital` to avoid dust,
   - or adjust trade sizing / exposure so less capital is tied up per trade.
