#!/bin/sh
exec npx ts-node -P tsconfig.worker.json scripts/runReachAprilBtcReactionMonitor.ts
