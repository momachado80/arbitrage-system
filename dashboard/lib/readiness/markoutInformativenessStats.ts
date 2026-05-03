/**
 * Métricas de qualidade/informatividade de markouts (paper-only, read-only).
 *
 * Distinção entre fechamento técnico completo vs evidência económica útil nos preços de follow-up.
 */

import fs from "fs";

import type {
  MarkoutInformativenessStats,
  MarkoutInformativenessVerdict,
} from "./microcapitalReadinessDossier";
import type { PaperCycleLifecycleSummary } from "./paperCycleLifecycle";

export type { MarkoutInformativenessStats, MarkoutInformativenessVerdict };

const MARKOUT_EPS = 1e-12;
const PINNED_REF = 0.001;
const PINNED_TOLERANCE = 1e-9;
export const MARKOUT_REQUIRED_HORIZONS_MS = [5_000, 30_000, 60_000] as const;

interface LooseFollowupRow {
  cycleId: string;
  horizonMs: number;
  samplerStatus: string | null;
  priceSource: string | null;
  markout: number | null;
  basePrice: number | null;
  followupPrice: number | null;
}

export function verdictFromInformativenessCounts(
  markoutBacked: number,
  informative: number,
  flat: number,
): MarkoutInformativenessVerdict {
  if (markoutBacked <= 0) return "INSUFFICIENT_MARKOUTS";
  if (informative <= 0) return "WEAK_FLAT_MARKOUTS";
  if (informative > flat) return "INFORMATIVE";
  if (informative === flat && informative > 0) return "INFORMATIVE";
  return "WEAK_FLAT_MARKOUTS";
}

function isPinnedLowPrice(basePrice: number | null, followupPrice: number | null): boolean {
  if (basePrice === null || followupPrice === null) return false;
  return (
    Math.abs(basePrice - PINNED_REF) <= PINNED_TOLERANCE &&
    Math.abs(followupPrice - PINNED_REF) <= PINNED_TOLERANCE
  );
}

function looseRowFromRecord(obj: Record<string, unknown>): LooseFollowupRow | null {
  if (obj.type !== "markout_followup") return null;
  if (obj.markoutStatus === "historical_followup_unavailable") return null;
  const cycleId = typeof obj.cycleId === "string" ? obj.cycleId : null;
  const horizonMs = typeof obj.horizonMs === "number" ? obj.horizonMs : null;
  if (!cycleId || horizonMs === null) return null;
  if (!MARKOUT_REQUIRED_HORIZONS_MS.includes(horizonMs as 5000 | 30000 | 60000)) return null;

  const statusRaw = typeof obj.samplerStatus === "string" ? obj.samplerStatus : null;
  const markout = typeof obj.markout === "number" ? obj.markout : null;
  const basePrice = typeof obj.basePrice === "number" ? obj.basePrice : null;
  const followupPrice = typeof obj.followupPrice === "number" ? obj.followupPrice : null;
  const priceSource = typeof obj.priceSource === "string" ? obj.priceSource : null;

  return {
    cycleId,
    horizonMs,
    samplerStatus: statusRaw,
    priceSource,
    markout,
    basePrice,
    followupPrice,
  };
}

function isSamplerOkRow(r: LooseFollowupRow): boolean {
  if (r.samplerStatus === "ok") {
    return r.markout !== null && r.basePrice !== null && r.followupPrice !== null;
  }
  const legacySamplerMissing = r.samplerStatus === null;
  if (
    legacySamplerMissing &&
    r.markout !== null &&
    r.basePrice !== null &&
    r.followupPrice !== null
  ) {
    return true;
  }
  return false;
}

/** @internal exported for tests */
export function readLooseMarkoutFollowupRowsSync(filePath: string): LooseFollowupRow[] {
  let raw: string;
  try {
    raw = fs.readFileSync(filePath, "utf8");
  } catch {
    return [];
  }
  const out: LooseFollowupRow[] = [];
  for (const line of raw.split(/\r?\n/)) {
    const t = line.trim();
    if (!t) continue;
    let obj: Record<string, unknown>;
    try {
      obj = JSON.parse(t) as Record<string, unknown>;
    } catch {
      continue;
    }
    const row = looseRowFromRecord(obj);
    if (row) out.push(row);
  }
  return out;
}

function horizonOkRow(
  byHorizon: Map<number, LooseFollowupRow>,
  hz: (typeof MARKOUT_REQUIRED_HORIZONS_MS)[number],
): LooseFollowupRow | undefined {
  const r = byHorizon.get(hz);
  if (!r || !isSamplerOkRow(r)) return undefined;
  return r;
}

/**
 * Agrupa loosely-parsed linhas por ciclo fechado (lifecycle) e calcula contagens seguras para dados legacy.
 */
export function computeMarkoutInformativenessStats(
  lifecycle: Pick<PaperCycleLifecycleSummary, "markoutBackedCycleCount" | "events">,
  followupsJsonlPath: string,
): MarkoutInformativenessStats | null {
  let rows: LooseFollowupRow[];
  try {
    if (!followupsJsonlPath || !fs.statSync(followupsJsonlPath).isFile()) return null;
    rows = readLooseMarkoutFollowupRowsSync(followupsJsonlPath);
  } catch {
    return null;
  }

  const closedCycleIds = new Set(
    lifecycle.events.filter(e => e.type === "paper_cycle_closed").map(e => e.cycleId),
  );

  let nonZeroFollowups = 0;
  let zeroFollowups = 0;
  for (const r of rows) {
    if (!isSamplerOkRow(r) || r.markout === null) continue;
    if (Math.abs(r.markout) > MARKOUT_EPS) nonZeroFollowups++;
    else zeroFollowups++;
  }

  let informativeCycles = 0;
  let flatCycles = 0;
  let allHorizonsFlat = 0;
  let lowPricePinnedCycles = 0;
  let bestAskOnlyCycles = 0;

  for (const cycleId of Array.from(closedCycleIds)) {
    const cycleRows = rows.filter(r => r.cycleId === cycleId);
    const byHorizon = new Map<number, LooseFollowupRow>();
    for (const r of cycleRows) {
      if (!isSamplerOkRow(r)) continue;
      byHorizon.set(r.horizonMs, r);
    }

    const h5 = horizonOkRow(byHorizon, 5_000);
    const h30 = horizonOkRow(byHorizon, 30_000);
    const h60 = horizonOkRow(byHorizon, 60_000);
    if (!h5 || !h30 || !h60) {
      flatCycles++;
      continue;
    }

    const trio = [h5, h30, h60];

    const allZeroMarkouts = trio.every(
      r => r.markout !== null && Math.abs(r.markout) <= MARKOUT_EPS,
    );

    const pricesSpread =
      trio.every(r => r.followupPrice !== null) &&
      (() => {
        const fp = trio.map(r => r.followupPrice!);
        const mx = Math.max(...fp);
        const mn = Math.min(...fp);
        return mx - mn > MARKOUT_EPS;
      })();

    const anyNonZero = trio.some(
      r => r.markout !== null && Math.abs(r.markout) > MARKOUT_EPS,
    );

    const informativeCycle = Boolean(anyNonZero || pricesSpread);

    if (informativeCycle) informativeCycles++;
    else flatCycles++;

    if (allZeroMarkouts) allHorizonsFlat++;

    const allPinnedLow = trio.every(r => isPinnedLowPrice(r.basePrice, r.followupPrice));
    if (allPinnedLow) lowPricePinnedCycles++;

    const allBestAsk = trio.every(
      r =>
        typeof r.priceSource === "string" &&
        String(r.priceSource).toLowerCase() === "best_ask_only",
    );
    if (allBestAsk) bestAskOnlyCycles++;
  }

  const verdict = verdictFromInformativenessCounts(
    lifecycle.markoutBackedCycleCount,
    informativeCycles,
    flatCycles,
  );

  return {
    markoutBackedCycleCount: lifecycle.markoutBackedCycleCount,
    informativeMarkoutCycleCount: informativeCycles,
    flatMarkoutCycleCount: flatCycles,
    nonZeroMarkoutFollowupCount: nonZeroFollowups,
    zeroMarkoutFollowupCount: zeroFollowups,
    allHorizonsFlatCycleCount: allHorizonsFlat,
    bestAskOnlyCycleCount: bestAskOnlyCycles,
    lowPricePinnedCycleCount: lowPricePinnedCycles,
    markoutInformativenessVerdict: verdict,
  };
}
