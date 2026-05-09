# Shadow Persistence — Validation Report

## 1. Code-Level Persistence Flow

### Where `SHADOW_PERSISTENCE_PATH` is read
- **File:** `dashboard/lib/shadowClosedTradePersistence.ts`
- **Function:** `getFilePath()` (internal)
- **Line:** `process.env.SHADOW_PERSISTENCE_PATH || process.env.DATA_PATH || process.cwd()`
- **Fallback:** `DATA_PATH` → `process.cwd()`

### Exact persisted filename
- `shadow-closed-trades.json`
- **Full path:** `{SHADOW_PERSISTENCE_PATH || DATA_PATH || cwd}/shadow-closed-trades.json`

### Where writes happen
- **File:** `dashboard/lib/shadowSimulationStore.ts`
- **Function:** `closeShadowTrade()` (lines ~317–324)
- **Flow:** After pushing a closed trade to `state.closedTrades`, builds `byProfile` from all `profileStates`, calls `persistClosedTrades(byProfile)`
- **Persistence:** `shadowClosedTradePersistence.ts` → `persistClosedTrades()` → `writeFileSync(tmpPath)` + `renameSync(tmpPath, filePath)` (atomic write)
- **Throttle:** 5 seconds between writes

### Where rehydration happens
- **File:** `dashboard/lib/shadowSimulationService.ts`
- **Function:** `ensureShadowSimulation()` (line 653)
- **Order:** `rehydrateFromPersistence()` is called first, before `ensureMarketDataRunning()` and profile init
- **Implementation:** `shadowSimulationStore.ts` → `rehydrateFromPersistence()` → `restoreClosedTrades()` → loads snapshot, merges into `profileStates` for profiles in `SHADOW_PROFILES`

### Endpoints exposing persistence status
- `/api/shadow/audit` — `persistence` block
- `/api/shadow/adaptive` — `persistence` block

**Fields:**
- `persistedHistoryAvailable` (boolean)
- `persistedClosedTradesCount` (number)
- `rehydratedAt` (string | null)
- `persistenceMode` (string: "file")
- `persistencePath` (string) — full path used for the file

---

## 2. Runtime Validation Commands

### Run locally (no Railway)
```bash
node dashboard/scripts/validate-persistence-runtime.js
```

### Run in Railway environment (production)
```bash
railway link   # if not already linked
railway run node dashboard/scripts/validate-persistence-runtime.js
```

### Manual checks (if you have Railway shell / exec)
```bash
echo "SHADOW_PERSISTENCE_PATH=$SHADOW_PERSISTENCE_PATH"
echo "RAILWAY_VOLUME_MOUNT_PATH=$RAILWAY_VOLUME_MOUNT_PATH"
ls -la /data 2>/dev/null || echo "/data does not exist"
test -d /data && echo "/data is mounted" || echo "/data not found"
echo "test" > /data/.test 2>/dev/null && cat /data/.test && rm /data/.test
ls /data/shadow-closed-trades.json 2>/dev/null || echo "File not found"
head -c 500 /data/shadow-closed-trades.json 2>/dev/null || true
```

---

## 3. Production API Results (2026-03-12)

### GET /api/shadow/audit
```json
"persistence": {
  "persistedHistoryAvailable": false,
  "persistedClosedTradesCount": 0,
  "rehydratedAt": "2026-03-12T00:27:51.399Z",
  "persistenceMode": "file"
}
```
- `totalClosed`: 0 for all profiles
- `dataSufficiency.totalClosed`: 0

### GET /api/shadow/adaptive
```json
"persistence": {
  "persistedHistoryAvailable": false,
  "persistedClosedTradesCount": 0,
  "rehydratedAt": "2026-03-12T00:27:51.399Z",
  "persistenceMode": "file"
}
```

---

## 4. Conclusions

| Question | Answer |
|---------|--------|
| **Is the Railway volume objectively mounted?** | Unknown without shell access. `/data` is not observable from API. |
| **Do writes to /data work?** | Unknown. `persistedHistoryAvailable: false` suggests either no volume, no `SHADOW_PERSISTENCE_PATH`, or no closed trades yet. |
| **Does persisted history survive redeploy?** | Not testable yet — no persisted history exists in production. |
| **Do audit/adaptive reflect persisted history?** | They would if a file existed; currently `totalClosed: 0` and `persistedClosedTradesCount: 0`. |

### Likely cause of `persistedHistoryAvailable: false`
1. `SHADOW_PERSISTENCE_PATH` not set → path is `process.cwd()` (ephemeral)
2. No Railway Volume at `/data` → even if set, `/data` may not exist
3. No closed trades yet → file never created

### Required for full validation
1. Set `SHADOW_PERSISTENCE_PATH=/data` in Railway Variables
2. Add a Volume mounted at `/data`
3. Run `railway run node dashboard/scripts/validate-persistence-runtime.js` to confirm `/data` exists and is writable
4. Let shadow simulation close trades, then redeploy and re-check endpoints
