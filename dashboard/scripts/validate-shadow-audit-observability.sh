#!/bin/sh
# Validate shadow audit observability fields are present in production.
# Usage: BASE_URL=https://your-dashboard.railway.app ./scripts/validate-shadow-audit-observability.sh
# Or: ./scripts/validate-shadow-audit-observability.sh https://dashboard-next-production-b126.up.railway.app

BASE="${BASE_URL:-${1:-https://dashboard-next-production-b126.up.railway.app}}"
BASE="${BASE%%/}"

required="byCapturableEdgeDecile byFillRatioBucket worstPairs exitReasonDiagnostics profileComparison causalReadout"

audit=$(curl -sf --max-time 15 "${BASE}/api/shadow/audit") || { echo "ERR: Failed to fetch audit"; exit 1; }
version=$(curl -sf --max-time 5 "${BASE}/api/version") || version=""

for k in $required; do
  if echo "$audit" | grep -q "\"${k}\""; then
    echo "OK: $k"
  else
    echo "MISSING: $k"
  fi
done

sha=$(echo "$version" | grep -o '"gitCommitShaShort"[[:space:]]*:[[:space:]]*"[^"]*"' | cut -d'"' -f4)
echo ""
echo "Deploy: $sha"
echo "URL: $BASE"
