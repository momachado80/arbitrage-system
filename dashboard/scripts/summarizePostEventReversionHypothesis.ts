/**
 * Post-Event Reversion — summarizer read-only.
 *
 * Lê o ledger JSONL append-only produzido por runPostEventReversionScout,
 * agrupa snapshots por (marketId, catalystEventStartUtc), monta EventSnapshot
 * por evento, computa ReversionMetric e julga via judgeHypothesis.
 *
 * Sem trade. Sem .paper. Sem chamada de execução. Apenas leitura + estatística.
 */

import fs from "fs";
import path from "path";

import {
  computeReversionMetric,
  judgeHypothesis,
  HYPOTHESIS_VERSION,
  type EventSnapshot,
  type EventSnapshotsByWindow,
  type SnapshotData,
  type HypothesisSport,
} from "../lib/postEventReversionHypothesis";

const DEFAULT_LEDGER = path.join(
  process.env.HOME ?? ".",
  "post-event-reversion-history.jsonl",
);
const LEDGER_PATH = process.env.POST_EVENT_REVERSION_LEDGER_PATH ?? DEFAULT_LEDGER;

interface LedgerEntry {
  timestamp: string;
  hypothesisVersion: string;
  marketId: string;
  question: string;
  sport: HypothesisSport;
  catalystEventStartUtc: string;
  windowType: string;
  windowRunAtUtc: string;
  snapshot: SnapshotData;
}

function readLedger(p: string): LedgerEntry[] {
  if (!fs.existsSync(p)) return [];
  const raw = fs.readFileSync(p, "utf8");
  const out: LedgerEntry[] = [];
  for (const line of raw.split(/\r?\n/)) {
    if (!line.trim()) continue;
    try {
      const j = JSON.parse(line) as LedgerEntry;
      if (j.hypothesisVersion !== HYPOTHESIS_VERSION) continue;
      out.push(j);
    } catch {
      /* ignore */
    }
  }
  return out;
}

function buildEvents(entries: LedgerEntry[]): EventSnapshot[] {
  /** Agrupa por (marketId, catalystEventStartUtc). Para cada grupo, mantém o
   *  snapshot mais recente por windowType. */
  type Bucket = Map<string, { snap: SnapshotData; recordedAt: string }>;
  const groups = new Map<string, { marketId: string; question: string; sport: HypothesisSport; catalystEventStartUtc: string; byWindow: Bucket }>();
  for (const e of entries) {
    const key = `${e.marketId}|${e.catalystEventStartUtc}`;
    let g = groups.get(key);
    if (!g) {
      g = {
        marketId: e.marketId,
        question: e.question,
        sport: e.sport,
        catalystEventStartUtc: e.catalystEventStartUtc,
        byWindow: new Map<string, { snap: SnapshotData; recordedAt: string }>(),
      };
      groups.set(key, g);
    }
    const prev = g.byWindow.get(e.windowType);
    if (!prev || e.timestamp > prev.recordedAt) {
      g.byWindow.set(e.windowType, { snap: e.snapshot, recordedAt: e.timestamp });
    }
  }
  const events: EventSnapshot[] = [];
  for (const g of Array.from(groups.values())) {
    const snapshots: EventSnapshotsByWindow = {};
    const pre = g.byWindow.get("PRE_EVENT_15M");
    const postImm = g.byWindow.get("POST_EVENT_15M");
    const postLate = g.byWindow.get("POST_EVENT_60M");
    if (pre) snapshots.preEvent15m = pre.snap;
    if (postImm) snapshots.postEvent15m = postImm.snap;
    if (postLate) snapshots.postEvent60m = postLate.snap;
    events.push({
      marketId: g.marketId,
      question: g.question,
      sport: g.sport,
      catalystEventStartUtc: g.catalystEventStartUtc,
      snapshots,
    });
  }
  return events;
}

function main(): void {
  const entries = readLedger(LEDGER_PATH);
  if (entries.length === 0) {
    process.stdout.write(
      `[post-event-reversion-summarize] ledger empty or missing at ${LEDGER_PATH}\n`,
    );
    process.exit(0);
  }
  const events = buildEvents(entries);
  const metrics = events.map(computeReversionMetric);
  const verdict = judgeHypothesis(metrics);

  /** Breakdown por sport para análise. */
  const bySport: Record<string, { qualified: number; mean: number | null }> = {};
  for (const sport of ["NBA", "NHL"] as const) {
    const sportMetrics = metrics.filter(m => m.sport === sport);
    const sportQualified = sportMetrics.filter(
      m => m.signalFired && m.realizedReversion !== null,
    );
    const n = sportQualified.length;
    const mean = n > 0
      ? sportQualified.reduce((a, b) => a + (b.realizedReversion ?? 0), 0) / n
      : null;
    bySport[sport] = { qualified: n, mean: mean !== null ? Math.round(mean * 1e6) / 1e6 : null };
  }

  const summary = {
    hypothesisVersion: HYPOTHESIS_VERSION,
    ledgerPath: LEDGER_PATH,
    eventsTotal: events.length,
    metricsTotal: metrics.length,
    verdict,
    bySport,
    invalidationBreakdown: tallyInvalidations(metrics),
  };
  process.stdout.write(`${JSON.stringify(summary, null, 2)}\n`);
}

function tallyInvalidations(
  metrics: ReturnType<typeof computeReversionMetric>[],
): Record<string, number> {
  const out: Record<string, number> = {};
  for (const m of metrics) {
    const key = m.invalidationReason ?? "qualified";
    out[key] = (out[key] ?? 0) + 1;
  }
  return out;
}

main();
