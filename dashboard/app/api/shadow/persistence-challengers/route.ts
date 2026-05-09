/**
 * Shadow Persistence Challengers Diagnostic — post-mortem only.
 * Reads raw persistence file and returns challenger (non-baseline) closed trades.
 * Rehydration skips challengers; this endpoint exposes them for analysis.
 * Does NOT change trading logic.
 */

import { NextResponse } from "next/server";
import { restoreClosedTrades } from "@/lib/shadowClosedTradePersistence";
import { SHADOW_PROFILES } from "@/lib/shadowSimulationProfiles";
import type { ShadowTrade } from "@/lib/shadowSimulationStore";
import type { ClosedTradeAuditEntry } from "@/lib/shadowClosedTradeAudit";

const BASELINE_IDS = new Set(SHADOW_PROFILES.map((p) => p.profileId));

function toAuditEntry(t: ShadowTrade, profileId: string): ClosedTradeAuditEntry {
  const holdingMs = t.holdingTimeMs ?? 0;
  const fc = t.filledCapital ?? 0;
  const rc = t.requestedCapital;
  const fillRatio = t.fillRatio ?? (rc != null && rc > 0 ? fc / rc : null);
  const mk = t.marketsInvolved;
  const pairKey =
    t.pairKey ?? (mk?.length ? [...(mk.map((x) => x.marketId).filter(Boolean) as string[])].sort().join("+") : null) ?? null;
  const bucket = holdingMs < 60_000 ? "<1min" : holdingMs < 300_000 ? "1-5min" : "5-15min";
  return {
    tradeId: t.tradeId,
    profileId,
    opportunityId: t.opportunityId ?? "",
    opportunityType: t.opportunityType ?? "unknown",
    sourceType: t.sourceType ?? "standard",
    exitReason: t.exitReason ?? "unknown",
    filledCapital: fc,
    realizedPnL: t.realizedPnL ?? 0,
    realizedReturn: t.realizedReturn ?? 0,
    holdingTimeMs: holdingMs,
    holdingTimeBucket: bucket,
    observedEdgeAtEntry: t.observedEdgeAtEntry ?? 0,
    capturableEdgeAtEntry: t.capturableEdgeAtEntry ?? 0,
    effectiveEntryPrice: t.effectiveEntryPrice ?? 0,
    effectiveExitPrice: t.effectiveExitPrice ?? 0,
    openedAt: t.openedAt ?? "",
    closedAt: t.closedAt ?? "",
    fillRatio,
    pairKey,
  };
}

function median(arr: number[]): number {
  if (arr.length === 0) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  const m = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[m] : (sorted[m - 1] + sorted[m]) / 2;
}

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const snapshot = restoreClosedTrades();
    if (!snapshot?.byProfile) {
      return NextResponse.json({
        challengerProfileIds: [],
        byProfile: {},
        message: "No persistence data or empty snapshot.",
      });
    }

    const challengerIds = Object.keys(snapshot.byProfile).filter((id) => !BASELINE_IDS.has(id));
    const byProfile: Record<
      string,
      {
        entries: ClosedTradeAuditEntry[];
        totalClosed: number;
        avgRealizedPnL: number;
        medianRealizedPnL: number;
        totalRealizedPnL: number;
        avgHoldingTimeMs: number;
        avgFilledCapital: number;
        avgFillRatio: number;
        avgObservedEdgeAtEntry: number;
        avgCapturableEdgeAtEntry: number;
        avgEffectiveEntryPrice: number;
        byExitReason: Record<string, { count: number; totalPnL: number }>;
        byPairKey: Record<string, { count: number; totalPnL: number }>;
        byCapturableEdgeDecile: Record<string, { count: number; avgPnL: number; avgFilledCapital: number }>;
        byFilledCapitalDecile: Record<string, { count: number; avgPnL: number }>;
        worst10: ClosedTradeAuditEntry[];
        best10: ClosedTradeAuditEntry[];
      }
    > = {};

    for (const profileId of challengerIds) {
      const raw = snapshot.byProfile[profileId] ?? [];
      const closed = raw.filter((t): t is ShadowTrade => !!(t?.tradeId && t?.status === "closed" && t?.closedAt));
      const entries = closed.map((t) => toAuditEntry(t, profileId));

      const byExit: Record<string, { count: number; totalPnL: number }> = {};
      const byPair: Record<string, { count: number; totalPnL: number }> = {};
      for (const e of entries) {
        byExit[e.exitReason] = byExit[e.exitReason] ?? { count: 0, totalPnL: 0 };
        byExit[e.exitReason].count++;
        byExit[e.exitReason].totalPnL += e.realizedPnL;
        const pk = e.pairKey ?? "unknown";
        byPair[pk] = byPair[pk] ?? { count: 0, totalPnL: 0 };
        byPair[pk].count++;
        byPair[pk].totalPnL += e.realizedPnL;
      }

      const byEdgeDecile: Record<string, { count: number; avgPnL: number; avgFilledCapital: number }> = {};
      const withEdge = entries.filter((e) => e.capturableEdgeAtEntry != null);
      const sortedEdge = [...withEdge].sort((a, b) => a.capturableEdgeAtEntry - b.capturableEdgeAtEntry);
      for (let d = 1; d <= 10; d++) {
        const start = Math.floor(((d - 1) / 10) * sortedEdge.length);
        const end = Math.floor((d / 10) * sortedEdge.length);
        const slice = sortedEdge.slice(start, end);
        if (slice.length) {
          byEdgeDecile[`d${d}`] = {
            count: slice.length,
            avgPnL: slice.reduce((s, e) => s + e.realizedPnL, 0) / slice.length,
            avgFilledCapital: slice.reduce((s, e) => s + e.filledCapital, 0) / slice.length,
          };
        }
      }

      const byFcDecile: Record<string, { count: number; avgPnL: number }> = {};
      const sortedFc = [...entries].sort((a, b) => a.filledCapital - b.filledCapital);
      for (let d = 1; d <= 10; d++) {
        const start = Math.floor(((d - 1) / 10) * sortedFc.length);
        const end = Math.floor((d / 10) * sortedFc.length);
        const slice = sortedFc.slice(start, end);
        if (slice.length) {
          byFcDecile[`d${d}`] = {
            count: slice.length,
            avgPnL: slice.reduce((s, e) => s + e.realizedPnL, 0) / slice.length,
          };
        }
      }

      const pnls = entries.map((e) => e.realizedPnL);
      const sortedPnL = [...entries].sort((a, b) => a.realizedPnL - b.realizedPnL);
      const fillRatios = entries.map((e) => e.fillRatio ?? 0).filter((r) => r > 0);

      byProfile[profileId] = {
        entries,
        totalClosed: entries.length,
        avgRealizedPnL: entries.length ? entries.reduce((s, e) => s + e.realizedPnL, 0) / entries.length : 0,
        medianRealizedPnL: median(pnls),
        totalRealizedPnL: entries.reduce((s, e) => s + e.realizedPnL, 0),
        avgHoldingTimeMs: entries.length ? entries.reduce((s, e) => s + e.holdingTimeMs, 0) / entries.length : 0,
        avgFilledCapital: entries.length ? entries.reduce((s, e) => s + e.filledCapital, 0) / entries.length : 0,
        avgFillRatio: fillRatios.length ? fillRatios.reduce((a, b) => a + b, 0) / fillRatios.length : 0,
        avgObservedEdgeAtEntry: entries.length ? entries.reduce((s, e) => s + e.observedEdgeAtEntry, 0) / entries.length : 0,
        avgCapturableEdgeAtEntry: entries.length ? entries.reduce((s, e) => s + e.capturableEdgeAtEntry, 0) / entries.length : 0,
        avgEffectiveEntryPrice: entries.length ? entries.reduce((s, e) => s + e.effectiveEntryPrice, 0) / entries.length : 0,
        byExitReason: byExit,
        byPairKey: byPair,
        byCapturableEdgeDecile: byEdgeDecile,
        byFilledCapitalDecile: byFcDecile,
        worst10: sortedPnL.slice(0, 10),
        best10: sortedPnL.slice(-10).reverse(),
      };
    }

    return NextResponse.json({
      snapshotSavedAt: snapshot.savedAt,
      challengerProfileIds: challengerIds,
      byProfile,
    });
  } catch (err) {
    console.error("[API /shadow/persistence-challengers]", err);
    return NextResponse.json(
      { error: "Failed to read challenger data", message: (err as Error).message },
      { status: 500 }
    );
  }
}
