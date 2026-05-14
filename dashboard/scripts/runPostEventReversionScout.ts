/**
 * Post-Event Reversion Scout — runner read-only puro para a hipótese narrow #1.
 *
 * Lê um plano de catalisadores (já produzido por buildCatalystObservationPlan)
 * e a cada ciclo:
 *  1. filtra entradas elegíveis (NBA Finals + NHL Stanley Cup championship)
 *  2. seleciona janelas-alvo (PRE_EVENT_15M, POST_EVENT_15M, POST_EVENT_60M)
 *  3. para cada janela due dentro da tolerância, dedup contra ledger e
 *     appenda snapshot read-only (mid/spread/liquidity/volume da Gamma) em JSONL
 *
 * Sem trade. Sem chamada a executionDispatcher. Sem paperPortfolioStore.
 * Sem shadowSimulationStore. Sem worker de execução. Sem caminhos de
 * envio para rede externa além do GET de mercados na Gamma. Sem .paper.
 *
 * Pré-requisito de produção:
 *   npx ts-node scripts/buildCatalystObservationPlan.ts --out /tmp/catalyst-observation-plan.json
 *
 * Variáveis de ambiente:
 *   POST_EVENT_REVERSION_LEDGER_PATH    (default: $HOME/post-event-reversion-history.jsonl)
 *   POST_EVENT_REVERSION_PLAN_PATH      (default: /tmp/catalyst-observation-plan.json)
 *   POST_EVENT_REVERSION_POLL_INTERVAL_SECONDS  (default: 300)
 *   POST_EVENT_REVERSION_DUE_TOLERANCE_MINUTES  (default: 8)
 *   POST_EVENT_REVERSION_AUTOMATION_ENABLED     (default: 0 — exige --once ou set 1)
 */

import fs from "fs";
import path from "path";

import { fetchAllMarkets, type NormalizedMarket } from "../lib/polymarketClient";
import {
  HYPOTHESIS_VERSION,
  type SnapshotData,
} from "../lib/postEventReversionHypothesis";
import {
  buildDedupeKey,
  collectDueTargets,
  type PlanFile,
} from "../lib/postEventReversionPlanReader";

const POLL_INTERVAL_SECONDS = parseInt(
  process.env.POST_EVENT_REVERSION_POLL_INTERVAL_SECONDS ?? "300",
  10,
);
const DUE_TOLERANCE_MINUTES = parseInt(
  process.env.POST_EVENT_REVERSION_DUE_TOLERANCE_MINUTES ?? "8",
  10,
);
const AUTOMATION_ENABLED =
  process.env.POST_EVENT_REVERSION_AUTOMATION_ENABLED === "1";
const DEFAULT_LEDGER = path.join(
  process.env.HOME ?? ".",
  "post-event-reversion-history.jsonl",
);
const LEDGER_PATH =
  process.env.POST_EVENT_REVERSION_LEDGER_PATH ?? DEFAULT_LEDGER;
const PLAN_PATH =
  process.env.POST_EVENT_REVERSION_PLAN_PATH ?? "/tmp/catalyst-observation-plan.json";

function parseArgv(argv: string[]): { once: boolean } {
  return { once: argv.includes("--once") };
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/** Guard simples: o ledger jamais entra em .paper. */
function assertSafeLedgerPath(p: string): void {
  const norm = path.resolve(p);
  if (
    norm.includes(`${path.sep}.paper${path.sep}`) ||
    norm.endsWith(`${path.sep}.paper`)
  ) {
    throw new Error("ledger_path_blocked:.paper");
  }
}

function readSampledKeys(ledgerPath: string, maxLines = 8000): Set<string> {
  const seen = new Set<string>();
  if (!fs.existsSync(ledgerPath)) return seen;
  try {
    const raw = fs.readFileSync(ledgerPath, "utf8");
    const lines = raw.split(/\r?\n/).filter(Boolean);
    const slice = lines.length > maxLines ? lines.slice(-maxLines) : lines;
    for (const line of slice) {
      try {
        const j = JSON.parse(line) as Record<string, unknown>;
        const k = j.dedupeKey;
        if (typeof k === "string" && k.length > 0) seen.add(k);
      } catch {
        /* linha inválida — ignorar */
      }
    }
  } catch {
    /* não-fatal */
  }
  return seen;
}

function snapshotFromMarket(
  market: NormalizedMarket,
  capturedAtUtc: string,
): SnapshotData {
  /** NormalizedMarket não traz bestBid/bestAsk; deixamos null. spread/liquidity/
   *  volume vêm direto do payload Gamma. */
  const mid = market.prices.length > 0 ? market.prices[0]! : 0;
  return {
    capturedAtUtc,
    mid,
    bid: null,
    ask: null,
    spread: market.spread,
    liquidity: market.liquidity,
    volume: market.volume,
  };
}

function appendLedger(ledgerPath: string, entry: Record<string, unknown>): void {
  const dir = path.dirname(path.resolve(ledgerPath));
  fs.mkdirSync(dir, { recursive: true });
  fs.appendFileSync(
    path.resolve(ledgerPath),
    `${JSON.stringify(entry)}\n`,
    "utf8",
  );
}

async function runOneCycle(): Promise<void> {
  const now = new Date();
  const capturedAt = now.toISOString();

  let plan: PlanFile;
  try {
    const raw = fs.readFileSync(path.resolve(PLAN_PATH), "utf8");
    plan = JSON.parse(raw) as PlanFile;
  } catch (err) {
    process.stderr.write(
      `[post-event-reversion-scout] plan_read_failed at ${PLAN_PATH}: ${err instanceof Error ? err.message : String(err)}\n`,
    );
    return;
  }

  const markets = await fetchAllMarkets();
  const marketsById = new Map(markets.map(m => [m.id, m]));
  const targets = collectDueTargets(plan, marketsById, capturedAt, now, DUE_TOLERANCE_MINUTES);
  const seen = readSampledKeys(LEDGER_PATH);
  const fresh = targets.filter(
    t => !seen.has(buildDedupeKey(t.marketId, t.windowType, t.runAtUtc)),
  );

  for (const target of fresh) {
    const snap = snapshotFromMarket(target.market, capturedAt);
    const dedupeKey = buildDedupeKey(
      target.marketId,
      target.windowType,
      target.runAtUtc,
    );
    const entry = {
      timestamp: capturedAt,
      hypothesisVersion: HYPOTHESIS_VERSION,
      marketId: target.marketId,
      question: target.question,
      sport: target.sport,
      catalystEventStartUtc: target.catalystEventStartUtc,
      windowType: target.windowType,
      windowRunAtUtc: target.runAtUtc,
      snapshot: snap,
      dedupeKey,
    };
    appendLedger(LEDGER_PATH, entry);
    process.stdout.write(
      `[post-event-reversion-scout] sampled marketId=${target.marketId} window=${target.windowType} sport=${target.sport} mid=${snap.mid}\n`,
    );
  }

  process.stdout.write(
    `[post-event-reversion-scout] ${capturedAt} markets=${markets.length} dueTargets=${targets.length} fresh=${fresh.length}\n`,
  );
}

async function main(): Promise<void> {
  assertSafeLedgerPath(LEDGER_PATH);
  const { once } = parseArgv(process.argv);

  if (!once && !AUTOMATION_ENABLED) {
    process.stderr.write(
      "[post-event-reversion-scout] disabled — set POST_EVENT_REVERSION_AUTOMATION_ENABLED=1 ou use --once.\n",
    );
    process.exit(0);
  }

  process.stdout.write(
    `[post-event-reversion-scout] starting once=${once} pollSeconds=${POLL_INTERVAL_SECONDS} ledger=${LEDGER_PATH} plan=${PLAN_PATH}\n`,
  );

  let shuttingDown = false;
  process.on("SIGINT", () => {
    shuttingDown = true;
    process.stderr.write("\n[post-event-reversion-scout] SIGINT → graceful stop\n");
  });
  process.on("SIGTERM", () => {
    shuttingDown = true;
    process.stderr.write("\n[post-event-reversion-scout] SIGTERM → graceful stop\n");
  });

  do {
    try {
      await runOneCycle();
    } catch (err) {
      process.stderr.write(
        `[post-event-reversion-scout] cycle error: ${err instanceof Error ? err.message : String(err)}\n`,
      );
    }
    if (once || shuttingDown) break;
    await sleep(POLL_INTERVAL_SECONDS * 1000);
  } while (!shuttingDown);

  process.stdout.write("[post-event-reversion-scout] exit_ok\n");
}

main().catch(err => {
  process.stderr.write(
    `[post-event-reversion-scout] fatal: ${err instanceof Error ? err.message : String(err)}\n`,
  );
  process.exitCode = 1;
});
