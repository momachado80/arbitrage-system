#!/bin/sh
# Validate that a URL points to the Next.js dashboard (hosts /api/deployment-info).
# Usage: ./scripts/validate-dashboard-url.sh <url>
# Exit 0 + prints dashboardUrl if valid; exit 1 otherwise.

URL="${1:?Usage: $0 <url>}"
BASE="${URL%%/}"
RES="${BASE}/api/deployment-info"

resp=$(curl -sf --max-time 10 "${RES}" 2>/dev/null) || exit 1
url=$(echo "$resp" | grep -o '"dashboardUrl"[[:space:]]*:[[:space:]]*"[^"]*"' | cut -d'"' -f4)
success=$(echo "$resp" | grep -o '"success"[[:space:]]*:[[:space:]]*true')

if [ -n "$success" ] && [ -n "$url" ]; then
  echo "$url"
  exit 0
fi
exit 1
