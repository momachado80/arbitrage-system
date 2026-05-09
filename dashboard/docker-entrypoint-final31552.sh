#!/bin/sh
# Railway pode anexar um startCommand herdado do railway.json da raiz; esta imagem ignora argumentos extra.
exec npx ts-node -P tsconfig.worker.json scripts/runFinalNegativeRisk31552.ts
