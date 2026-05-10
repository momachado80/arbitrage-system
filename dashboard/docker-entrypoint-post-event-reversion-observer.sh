#!/bin/sh
# Entrypoint do serviço Railway "post-event-reversion-observer".
# Loop sequencial single-process:
#  1. rebuild do plano de catalisadores via buildCatalystObservationPlan
#  2. scout read-only via runPostEventReversionScout --once
#  3. sleep por POST_EVENT_REVERSION_POLL_INTERVAL_SECONDS
#
# Read-only puro: sem trade, sem paper engine, sem .paper, sem microcapital,
# sem worker de execução, sem caminhos de envio para rede de execução real.
# Apenas leituras na Gamma da Polymarket + ESPN public schedule, e escritas
# append-only no JSONL do volume /data.

set -u

POLL_SECS=${POST_EVENT_REVERSION_POLL_INTERVAL_SECONDS:-600}
PLAN_PATH=${POST_EVENT_REVERSION_PLAN_PATH:-/data/catalyst-observation-plan.json}
LEDGER_PATH=${POST_EVENT_REVERSION_LEDGER_PATH:-/data/post-event-reversion-history.jsonl}
LIMIT=${POST_EVENT_REVERSION_PLAN_LIMIT:-30}
HORIZON=${POST_EVENT_REVERSION_PLAN_HORIZON_HOURS:-72}

# Volume mount root
mkdir -p "$(dirname "$LEDGER_PATH")" "$(dirname "$PLAN_PATH")"

echo "[post-event-reversion-observer] starting"
echo "[post-event-reversion-observer] poll_secs=$POLL_SECS limit=$LIMIT horizon_hours=$HORIZON"
echo "[post-event-reversion-observer] plan_path=$PLAN_PATH ledger_path=$LEDGER_PATH"

shutting_down=0
on_term() {
  echo "[post-event-reversion-observer] shutdown signal received, finishing current iteration"
  shutting_down=1
}
trap on_term INT TERM

while [ $shutting_down -eq 0 ]; do
  cycle_started=$(date -u +%FT%TZ)
  echo "[post-event-reversion-observer] cycle_start ts=$cycle_started"

  echo "[post-event-reversion-observer] step=build_plan"
  if npx ts-node -P tsconfig.worker.json scripts/buildCatalystObservationPlan.ts \
       --limit "$LIMIT" --horizon-hours "$HORIZON" --out "$PLAN_PATH"; then
    echo "[post-event-reversion-observer] build_plan ok"
  else
    echo "[post-event-reversion-observer] build_plan FAILED (continuing to scout with previous plan if any)"
  fi

  echo "[post-event-reversion-observer] step=scout_once"
  if POST_EVENT_REVERSION_PLAN_PATH="$PLAN_PATH" \
     POST_EVENT_REVERSION_LEDGER_PATH="$LEDGER_PATH" \
     npx ts-node -P tsconfig.worker.json scripts/runPostEventReversionScout.ts --once; then
    echo "[post-event-reversion-observer] scout_once ok"
  else
    echo "[post-event-reversion-observer] scout_once FAILED (will retry next cycle)"
  fi

  if [ $shutting_down -eq 1 ]; then break; fi
  echo "[post-event-reversion-observer] sleeping ${POLL_SECS}s"
  sleep "$POLL_SECS" &
  sleep_pid=$!
  wait $sleep_pid 2>/dev/null || true
done

echo "[post-event-reversion-observer] exit_ok"
