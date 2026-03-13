# Adaptive Challenger Activation

## Current experiment: shadow_1000_adapt_captrade_entryfloor_v1

**Single-variable experiment on top of captrade**: Add conservative entry-floor bump (minNetCapturableEdgeToTrade 0.007 → 0.009) only. Inherits maxCapitalPerTrade=75 from captrade v1. No edgegate, pair penalty, hold-time, exit, or sizing changes.

### Enabling the challenger

Set in **Railway → Your Service → Variables**:

```
ENABLED_ADAPTIVE_CHALLENGERS=shadow_1000_adapt_captrade_entryfloor_v1
```

- Remove shadow_1000_adapt_captrade_v1 (the new challenger replaces it for this experiment).
- A redeploy is required after changing Variables.

### Entry floor configuration

| Profile | minNetCapturableEdgeToTrade | maxCapitalPerTrade |
|---------|-----------------------------|---------------------|
| shadow_1000 (baseline) | 0.007 | 150 |
| shadow_1000_adapt_captrade_v1 | 0.007 | 75 |
| shadow_1000_adapt_captrade_entryfloor_v1 | 0.009 | 75 |

Increment of +0.002 is conservative (~28%); edgegate v1 used 0.012 and failed; 0.009 avoids aggressive gating.

### Verification

1. **GET** `https://dashboard-next-production-b126.up.railway.app/api/shadow/adaptive`  
   - `adaptiveChallengers` contains `shadow_1000_adapt_captrade_entryfloor_v1`  
   - `enabledForExecution: true` when env is set  
   - `baseProfileId: shadow_1000_adapt_captrade_v1`, `entryFloorOverride: 0.009`, `maxCapitalPerTradeOverride: 75`, `hypothesis`, `expectedMechanism` present  

2. **GET** `https://dashboard-next-production-b126.up.railway.app/api/shadow/profiles`  
   - Profile `shadow_1000_adapt_captrade_entryfloor_v1` appears with state

---

## Previous experiments (archived)

### shadow_1000_adapt_captrade_v1

```
ENABLED_ADAPTIVE_CHALLENGERS=shadow_1000_adapt_captrade_v1
```

50% cap-per-trade reduction (150 → 75). Reached 500+ closed trades; materially improved avgRealizedPnL vs shadow_1000; reduced avgFilledCapital. Base for entry-floor experiment.

### shadow_1000_adapt_pairpenalty_v1

```
ENABLED_ADAPTIVE_CHALLENGERS=shadow_1000_adapt_pairpenalty_v1
```

Penalized top 3 worst pairKeys (540820+573647, 573647+604490, 567561+604490) at 0.03 each.
