# Shadow Closed Trades Persistence (Railway)

To make shadow audit and adaptive survive redeploys:

1. In Railway Dashboard → your service → **Volumes** → Add a Volume.
2. Mount path: `/data`
3. In **Variables**, add: `SHADOW_PERSISTENCE_PATH=/data`

Closed shadow trades will then persist across redeploys. Without a volume, data is ephemeral and resets on each deploy.

**Trilha de observação (catalog-pocket / economics / execution / minimal paper / system-ladder):** no mesmo volume, define também `PAPER_STATE_DIR=/data` (ver `RAILWAY_OBSERVATION_TRAIL.md`).
