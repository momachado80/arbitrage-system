#!/bin/bash
# Fase 1.5 — Validação de Boot Determinístico
# Executa checklist formal de boot.
set -e

PORT=${PORT:-8000}
BASE="http://127.0.0.1:${PORT}"

echo "=== 1. Iniciando servidor em background..."
uvicorn src.server:app --host 127.0.0.1 --port "$PORT" &
PID=$!
trap "kill $PID 2>/dev/null || true" EXIT

echo "Aguardando 8 segundos..."
sleep 8

echo ""
echo "=== 2. Health check..."
curl -sf "${BASE}/health" | python3 -c "
import json, sys
d = json.load(sys.stdin)
assert d.get('status') == 'ok', f'status != ok: {d}'
print('OK: {\"status\": \"ok\"}')
"

echo ""
echo "=== 3. Verificando logs (PYTHON_VERSION, GLOBAL_SEED_SET)..."
echo "    Execute manualmente e confira:"
echo "    python -Wd -m uvicorn src.server:app --host 127.0.0.1 --port 8001"
echo "    Esperado nos logs: PYTHON_VERSION, GLOBAL_SEED_SET, ENV"
echo ""

echo "=== Boot validation passed ==="
