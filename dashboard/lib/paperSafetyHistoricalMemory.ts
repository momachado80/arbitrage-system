/**
 * Memória persistente mínima para safety gates operacionais (sobrevive a restart).
 * Ficheiro JSON em disco (por defeito `.paper/safety-class-memory.json`), actualizado
 * em cada fecho de trade — agregados por `sourceType|opportunityType` apenas.
 */

import fs from "fs";
import path from "path";
import type { PaperTrade } from "./paperTypes";
import {
  getClosedTradeGrossRealizedPnL,
  getClosedTradeNetRealizedPnL,
  DEFAULT_PAPER_FEE_BUFFER_PER_LEG,
} from "./paperRealizedPnlSemantics";
import { isClosedTradeWithFiniteRealizedPnl } from "./paperClosedTradesMetrics";
import { bumpPaperSafetyProfileCacheEpoch } from "./paperSafetyProfileCacheEpoch";

const MEMORY_VERSION = 1 as const;
const GROSS_ZERO_EPS = 0.01;

export type SafetyClassAggregateV1 = {
  closedCount: number;
  fallbackNoLatestCount: number;
  grossZeroNetNegCount: number;
};

export type SafetyMemoryFileV1 = {
  version: typeof MEMORY_VERSION;
  updatedAt: string;
  byClass: Record<string, SafetyClassAggregateV1>;
};

function defaultMemoryPath(): string {
  const raw = process.env.PAPER_SAFETY_MEMORY_PATH?.trim();
  if (raw && raw.length > 0) return path.resolve(raw);
  return path.join(process.cwd(), ".paper", "safety-class-memory.json");
}

function emptyAgg(): SafetyClassAggregateV1 {
  return { closedCount: 0, fallbackNoLatestCount: 0, grossZeroNetNegCount: 0 };
}

function readFileSafe(p: string): SafetyMemoryFileV1 | null {
  try {
    if (!fs.existsSync(p)) return null;
    const raw = fs.readFileSync(p, "utf8");
    const j = JSON.parse(raw) as Partial<SafetyMemoryFileV1>;
    if (j.version !== MEMORY_VERSION || typeof j.byClass !== "object" || j.byClass === null) {
      return null;
    }
    return j as SafetyMemoryFileV1;
  } catch {
    return null;
  }
}

function writeFileAtomic(p: string, data: SafetyMemoryFileV1): void {
  const dir = path.dirname(p);
  fs.mkdirSync(dir, { recursive: true });
  const tmp = `${p}.${process.pid}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(data, null, 0), "utf8");
  fs.renameSync(tmp, p);
}

export function getSafetyMemoryPathResolved(): string {
  return defaultMemoryPath();
}

export function loadSafetyClassMemory(): SafetyMemoryFileV1 {
  if (process.env.PAPER_SAFETY_DISABLE_DISK === "1") {
    return { version: MEMORY_VERSION, updatedAt: new Date().toISOString(), byClass: {} };
  }
  const p = defaultMemoryPath();
  const got = readFileSafe(p);
  if (got) return got;
  return { version: MEMORY_VERSION, updatedAt: new Date().toISOString(), byClass: {} };
}

/** Soma de closedCount em disco (para decidir backfill). */
export function totalPersistedCloseCount(mem: SafetyMemoryFileV1): number {
  let s = 0;
  for (const a of Object.values(mem.byClass)) {
    s += a.closedCount;
  }
  return s;
}

/**
 * Se o ficheiro está vazio mas o portfolio já tem fechados (ex.: antes da primeira persistência),
 * reconstrói agregados uma vez (auditorável nos logs).
 */
export function backfillSafetyMemoryFromClosedTrades(trades: PaperTrade[]): SafetyMemoryFileV1 {
  if (process.env.PAPER_SAFETY_DISABLE_DISK === "1") {
    return loadSafetyClassMemory();
  }
  const mem = loadSafetyClassMemory();
  if (totalPersistedCloseCount(mem) > 0) return mem;
  const closed = trades.filter(isClosedTradeWithFiniteRealizedPnl);
  if (closed.length === 0) return mem;

  const byClass: Record<string, SafetyClassAggregateV1> = {};
  const feeBuf = DEFAULT_PAPER_FEE_BUFFER_PER_LEG;
  for (const t of closed) {
    const k = `${t.sourceType}|${t.opportunityType}`;
    const a = byClass[k] ?? emptyAgg();
    a.closedCount += 1;
    if (t.exitPriceMarkSourceAtClose === "fallback_no_latest") a.fallbackNoLatestCount += 1;
    const gross = getClosedTradeGrossRealizedPnL(t);
    const net = getClosedTradeNetRealizedPnL(t, feeBuf);
    if (Math.abs(gross) < GROSS_ZERO_EPS && net < -GROSS_ZERO_EPS) a.grossZeroNetNegCount += 1;
    byClass[k] = a;
  }
  const out: SafetyMemoryFileV1 = {
    version: MEMORY_VERSION,
    updatedAt: new Date().toISOString(),
    byClass,
  };
  try {
    writeFileAtomic(defaultMemoryPath(), out);
    console.log(
      `[paperSafetyHistoricalMemory] backfill wrote ${closed.length} closes into ${defaultMemoryPath()}`
    );
    bumpPaperSafetyProfileCacheEpoch();
  } catch (e) {
    console.warn("[paperSafetyHistoricalMemory] backfill write failed:", e);
  }
  return out;
}

function classKeyFromTrade(t: PaperTrade): string {
  return `${t.sourceType}|${t.opportunityType}`;
}

/** Chamado em cada fecho — actualiza agregados persistentes. */
export function persistSafetyClose(trade: PaperTrade): void {
  if (process.env.PAPER_SAFETY_DISABLE_DISK === "1") return;
  if (trade.status !== "closed") return;
  if (!isClosedTradeWithFiniteRealizedPnl(trade)) return;

  const p = defaultMemoryPath();
  let mem = readFileSafe(p) ?? {
    version: MEMORY_VERSION,
    updatedAt: new Date().toISOString(),
    byClass: {} as Record<string, SafetyClassAggregateV1>,
  };

  const k = classKeyFromTrade(trade);
  const a = mem.byClass[k] ?? emptyAgg();
  a.closedCount += 1;
  if (trade.exitPriceMarkSourceAtClose === "fallback_no_latest") a.fallbackNoLatestCount += 1;
  const gross = getClosedTradeGrossRealizedPnL(trade);
  const net = getClosedTradeNetRealizedPnL(trade, DEFAULT_PAPER_FEE_BUFFER_PER_LEG);
  if (Math.abs(gross) < GROSS_ZERO_EPS && net < -GROSS_ZERO_EPS) a.grossZeroNetNegCount += 1;
  mem.byClass[k] = a;
  mem.updatedAt = new Date().toISOString();

  try {
    writeFileAtomic(p, mem);
    bumpPaperSafetyProfileCacheEpoch();
  } catch (e) {
    console.warn("[paperSafetyHistoricalMemory] persist write failed:", e);
  }
}
