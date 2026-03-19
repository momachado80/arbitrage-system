/**
 * Active Trade Aging Diagnostics — aging e estado de trades ativos por profile.
 * Responde: trades ainda abertos? há quanto tempo? edge atual? exit reason provável?
 */

import type { ShadowTrade } from "./shadowSimulationStore";
import type { ShadowProfileConfig } from "./shadowSimulationProfiles";

export interface ActiveTradeAgingEntry {
  tradeId: string;
  opportunityId: string;
  openedAt: string;
  holdingMs: number;
  entryEdge: number;
  currentObservedEdge: number | null;
  currentNetEdge: number | null;
  currentPnlPct: number | null;
  likelyExitReasonIfClosedNow: string;
  nearMaxHolding: boolean;
}

export interface ProfileActiveTradeAgingDiagnostics {
  profileId: string;
  label?: string;
  activeTradesCount: number;
  activeTradesAging: ActiveTradeAgingEntry[];
  oldestActiveTradeMs: number;
  avgActiveTradeMs: number;
  activeTradeIds: string[];
  avgObservedEdgeCurrent: number | null;
  avgNetEdgeCurrent: number | null;
  likelyExitReasonIfClosedNow: string;
  nearMaxHoldingCount: number;
  summary: string;
}

export interface ActiveTradeAgingDiagnostics {
  byProfile: Record<string, ProfileActiveTradeAgingDiagnostics>;
  generatedAt: string;
}

function inferLikelyExitReason(
  t: ShadowTrade,
  holdingMs: number,
  maxHoldingMs: number,
  stopLossPct: number,
  takeProfitPct: number,
  currentEdge: number | null
): string {
  if (holdingMs >= maxHoldingMs) return "max_holding_time";
  if (currentEdge == null) return "opportunity_absent_or_unknown";

  const exitPrice = 1 - currentEdge;
  const entryPrice = t.effectiveEntryPrice ?? 1 - (t.observedEdgeAtEntry ?? 0);
  const pnlPct = (exitPrice - entryPrice) / Math.max(0.001, entryPrice);

  if (pnlPct <= -stopLossPct) return "stop_loss";
  if (pnlPct >= takeProfitPct) return "take_profit";
  if (Math.abs(currentEdge) < 0.005) return "edge_normalization";

  return "within_hold_no_exit_trigger";
}

function buildProfileSummary(
  d: ProfileActiveTradeAgingDiagnostics,
  maxHoldingMs: number,
  profileId: string,
  closedTradesCount: number,
  funnel?: { finalCandidateCount: number; openedTradeCount: number }
): string {
  if (d.activeTradesCount === 0) {
    return "Nenhum trade ativo.";
  }

  const nearPct = d.oldestActiveTradeMs > 0
    ? Math.round((d.oldestActiveTradeMs / maxHoldingMs) * 100)
    : 0;

  if (d.nearMaxHoldingCount > 0 && nearPct >= 90) {
    return `Trades envelhecidos e presos: ${d.nearMaxHoldingCount} perto de max holding (${Math.round(maxHoldingMs / 60000)}min). Provável exit por max_holding_time no próximo ciclo.`;
  }

  if (d.likelyExitReasonIfClosedNow === "opportunity_absent_or_unknown" && d.activeTradesCount > 0) {
    return `Oportunidade ausente no merge atual: ${d.activeTradesCount} trade(s) ativo(s) sem edge observável. Pode indicar opp desapareceu ou anomalia operacional.`;
  }

  if (funnel && funnel.finalCandidateCount <= d.activeTradesCount && closedTradesCount === 0) {
    const lowCandidates =
      funnel.finalCandidateCount <= 5
        ? ` Baixa geração de novos candidates após os ${d.activeTradesCount} opens (finalCandidateCount=${funnel.finalCandidateCount}).`
        : "";
    if (d.oldestActiveTradeMs > maxHoldingMs * 0.5) {
      return `Trades ativos há ${Math.round(d.oldestActiveTradeMs / 60000)}min.${lowCandidates} Possível anomalia operacional ou regime com poucos candidates.`;
    }
  }

  if (d.oldestActiveTradeMs > maxHoldingMs * 0.7) {
    return `Trades dentro do regime mas próximos de max holding (${nearPct}%). ${d.likelyExitReasonIfClosedNow}.`;
  }

  if (d.oldestActiveTradeMs < maxHoldingMs * 0.3) {
    return `Trades ativos ainda dentro do regime normal. Mais antigo: ${Math.round(d.oldestActiveTradeMs / 60000)}min.`;
  }

  return `Trades ativos em regime intermediário. Mais antigo: ${Math.round(d.oldestActiveTradeMs / 60000)}min. ${d.likelyExitReasonIfClosedNow}.`;
}

export function getActiveTradeAgingDiagnostics(
  profiles: Array<{
    profileId: string;
    label?: string;
    activeTrades: ShadowTrade[];
    closedTradesCount?: number;
  }>,
  getProfileConfig: (profileId: string) => ShadowProfileConfig | undefined,
  currentOpportunitiesByOpportunityId: Map<string, { edge: number }>,
  funnelByProfile?: Record<string, { finalCandidateCount: number; openedTradeCount: number }>
): ActiveTradeAgingDiagnostics {
  const byProfile: Record<string, ProfileActiveTradeAgingDiagnostics> = {};
  const now = Date.now();

  for (const p of profiles) {
    const active = (p.activeTrades ?? []).filter((t) => t.status === "active");
    const cfg = getProfileConfig(p.profileId);
    const maxHoldingMs = cfg?.maxHoldingTimeMs ?? 300_000;
    const stopLossPct = cfg?.stopLossPct ?? 0.03;
    const takeProfitPct = cfg?.takeProfitPct ?? 0.05;

    const entries: ActiveTradeAgingEntry[] = [];
    let oldestMs = 0;
    let sumMs = 0;
    let nearMaxCount = 0;
    const observedEdges: number[] = [];
    const netEdges: number[] = [];
    let dominantExitReason = "within_hold_no_exit_trigger";

    for (const t of active) {
      const holdingMs = now - new Date(t.openedAt).getTime();
      oldestMs = Math.max(oldestMs, holdingMs);
      sumMs += holdingMs;

      const currentOpp = currentOpportunitiesByOpportunityId.get(t.opportunityId);
      const currentEdge = currentOpp?.edge ?? null;
      const currentObservedEdge = currentEdge;
      const currentNetEdge = currentEdge; // net ≈ observed para simplificar no diagnóstico
      let currentPnlPct: number | null = null;
      if (currentEdge != null && t.effectiveEntryPrice != null) {
        const exitPrice = 1 - currentEdge;
        currentPnlPct = (exitPrice - t.effectiveEntryPrice) / Math.max(0.001, t.effectiveEntryPrice);
      }

      const likelyReason = inferLikelyExitReason(
        t,
        holdingMs,
        maxHoldingMs,
        stopLossPct,
        takeProfitPct,
        currentEdge
      );
      const nearMax = holdingMs >= maxHoldingMs * 0.9;
      if (nearMax) nearMaxCount++;

      if (currentObservedEdge != null) observedEdges.push(currentObservedEdge);
      if (currentNetEdge != null) netEdges.push(currentNetEdge);
      if (dominantExitReason === "within_hold_no_exit_trigger" && likelyReason !== "within_hold_no_exit_trigger") {
        dominantExitReason = likelyReason;
      } else if (likelyReason === "max_holding_time" || likelyReason === "opportunity_absent_or_unknown") {
        dominantExitReason = likelyReason;
      }

      entries.push({
        tradeId: t.tradeId,
        opportunityId: t.opportunityId,
        openedAt: t.openedAt,
        holdingMs,
        entryEdge: t.observedEdgeAtEntry ?? 0,
        currentObservedEdge,
        currentNetEdge,
        currentPnlPct,
        likelyExitReasonIfClosedNow: likelyReason,
        nearMaxHolding: nearMax,
      });
    }

    const avgObserved =
      observedEdges.length > 0
        ? observedEdges.reduce((a, b) => a + b, 0) / observedEdges.length
        : null;
    const avgNet =
      netEdges.length > 0 ? netEdges.reduce((a, b) => a + b, 0) / netEdges.length : null;

    const d: ProfileActiveTradeAgingDiagnostics = {
      profileId: p.profileId,
      label: p.label,
      activeTradesCount: active.length,
      activeTradesAging: entries,
      oldestActiveTradeMs: oldestMs,
      avgActiveTradeMs: active.length > 0 ? sumMs / active.length : 0,
      activeTradeIds: active.map((t) => t.tradeId),
      avgObservedEdgeCurrent: avgObserved,
      avgNetEdgeCurrent: avgNet,
      likelyExitReasonIfClosedNow: dominantExitReason,
      nearMaxHoldingCount: nearMaxCount,
      summary: "",
    };
    const closedCount = p.closedTradesCount ?? 0;
    const funnel = funnelByProfile?.[p.profileId];
    d.summary = buildProfileSummary(d, maxHoldingMs, p.profileId, closedCount, funnel);
    byProfile[p.profileId] = d;
  }

  return {
    byProfile,
    generatedAt: new Date().toISOString(),
  };
}
