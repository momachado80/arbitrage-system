/**
 * Shadow Simulation Store — per-profile portfolio state, trades, analytics.
 * Bounded in memory; isolated per profile.
 */

import type { PaperSourceType, PaperOpportunityType } from "./paperTypes";
import type { ShadowProfileConfig } from "./shadowSimulationProfiles";

export interface ShadowTrade {
  tradeId: string;
  opportunityId: string;
  sourceType: PaperSourceType;
  opportunityType: PaperOpportunityType;
  clusterId?: string;
  marketsInvolved: Array<{ marketId: string; question: string }>;
  openedAt: string;
  closedAt: string | null;
  status: "active" | "closed";
  observedEdgeAtEntry: number;
  capturableEdgeAtEntry: number;
  effectiveEntryPrice: number;
  effectiveExitPrice?: number;
  filledCapital: number;
  requestedCapital?: number;
  realizedPnL: number;
  realizedReturn: number;
  holdingTimeMs: number;
  exitReason?: string;
  rejectionReason?: string;

  /** Diagnostic fields for audit (prospective for new trades; historical trades leave null) */
  fillRatio?: number | null;
  entryImpactBps?: number | null;
  exitImpactBps?: number | null;
  entryPriceModel?: string | null;
  exitPriceModel?: string | null;
  pairKey?: string | null;
  edgeAtExit?: number | null;
  edgeDecayDuringHold?: number | null;
  entryToExitPriceMove?: number | null;
  closeContext?: { exitReason?: string; [k: string]: unknown } | null;
}

export interface ShadowProfileState {
  profileId: string;
  label: string;
  startingCapital: number;
  currentEquity: number;
  availableCapital: number;
  reservedCapital: number;
  realizedPnL: number;
  unrealizedPnL: number;
  maxDrawdown: number;
  activeTrades: ShadowTrade[];
  closedTrades: ShadowTrade[];
  equityCurve: Array<{ ts: string; equity: number; cumulativePnL: number }>;
  lastUpdate: string | null;
  peakEquity: number;
  exposureByCluster: Record<string, number>;
  exposureByMarket: Record<string, number>;
}

export interface ShadowAnalytics {
  totalTrades: number;
  closedTrades: number;
  activeTrades: number;
  winRate: number;
  avgPnL: number;
  avgReturn: number;
  totalPnL: number;
  currentEquity: number;
  maxDrawdown: number;
  avgHoldingTimeMs: number;
  avgObservedEdgeAtEntry: number;
  avgCapturableEdgeAtEntry: number;
  avgFilledCapital: number;
  capacityUtilizationRate: number;
  rejectionRate: number;
  topRejectionReasons: Record<string, number>;
  pnlByOpportunityType: Record<string, number>;
  pnlBySourceType: Record<string, number>;
  edgeCaptureRatio: number;
  fillEfficiency: number;
  executionPenaltyEstimate: number;
}

const MAX_CLOSED_PER_PROFILE = 500;
const MAX_EQUITY_CURVE_POINTS = 200;
const rejectionCounts = new Map<string, Map<string, number>>();

const profileStates = new Map<string, ShadowProfileState>();

function initProfileState(config: ShadowProfileConfig): ShadowProfileState {
  return {
    profileId: config.profileId,
    label: config.label,
    startingCapital: config.startingCapital,
    currentEquity: config.startingCapital,
    availableCapital: config.startingCapital,
    reservedCapital: 0,
    realizedPnL: 0,
    unrealizedPnL: 0,
    maxDrawdown: 0,
    activeTrades: [],
    closedTrades: [],
    equityCurve: [{ ts: new Date().toISOString(), equity: config.startingCapital, cumulativePnL: 0 }],
    lastUpdate: null,
    peakEquity: config.startingCapital,
    exposureByCluster: {},
    exposureByMarket: {},
  };
}

export function ensureProfileState(config: ShadowProfileConfig): ShadowProfileState {
  let state = profileStates.get(config.profileId);
  if (!state) {
    state = initProfileState(config);
    profileStates.set(config.profileId, state);
  }
  return state;
}

export function getShadowProfileState(profileId: string): ShadowProfileState | null {
  return profileStates.get(profileId) ?? null;
}

export function getAllShadowProfiles(): ShadowProfileState[] {
  return Array.from(profileStates.values());
}

export function getShadowTrades(profileId: string): {
  active: ShadowTrade[];
  recentClosed: ShadowTrade[];
} {
  const state = profileStates.get(profileId);
  if (!state) return { active: [], recentClosed: [] };
  const start = Math.max(0, state.closedTrades.length - 50);
  return {
    active: [...state.activeTrades],
    recentClosed: state.closedTrades.slice(start),
  };
}

export function addShadowTrade(
  profileId: string,
  trade: ShadowTrade,
  config: ShadowProfileConfig
): void {
  try {
    const state = ensureProfileState(config);
    if (state.profileId !== profileId) return;
    state.activeTrades.push(trade);
    state.reservedCapital += trade.filledCapital;
    if (trade.clusterId) {
      state.exposureByCluster[trade.clusterId] = (state.exposureByCluster[trade.clusterId] ?? 0) + trade.filledCapital;
    }
    for (const m of trade.marketsInvolved ?? []) {
      state.exposureByMarket[m.marketId] = (state.exposureByMarket[m.marketId] ?? 0) + trade.filledCapital;
    }
    state.availableCapital = Math.max(0, state.currentEquity - state.reservedCapital);
    state.lastUpdate = new Date().toISOString();
  } catch {
    // non-fatal
  }
}

export function closeShadowTrade(
  profileId: string,
  tradeId: string,
  updates: Partial<ShadowTrade>
): void {
  try {
    const state = profileStates.get(profileId);
    if (!state) return;
    const idx = state.activeTrades.findIndex((t) => t.tradeId === tradeId);
    if (idx < 0) return;
    const t = state.activeTrades[idx];
    state.activeTrades.splice(idx, 1);
    const closed: ShadowTrade = {
      ...t,
      ...updates,
      status: "closed",
      closedAt: updates.closedAt ?? new Date().toISOString(),
    };
    state.closedTrades.push(closed);
    if (state.closedTrades.length > MAX_CLOSED_PER_PROFILE) {
      state.closedTrades = state.closedTrades.slice(-MAX_CLOSED_PER_PROFILE);
    }
    state.realizedPnL += closed.realizedPnL ?? 0;
    state.reservedCapital -= t.filledCapital;
    if (t.clusterId) {
      state.exposureByCluster[t.clusterId] = Math.max(0, (state.exposureByCluster[t.clusterId] ?? 0) - t.filledCapital);
    }
    for (const m of t.marketsInvolved ?? []) {
      state.exposureByMarket[m.marketId] = Math.max(0, (state.exposureByMarket[m.marketId] ?? 0) - t.filledCapital);
    }
    state.currentEquity = state.startingCapital + state.realizedPnL + state.unrealizedPnL;
    state.availableCapital = Math.max(0, state.currentEquity - state.reservedCapital);
    if (state.currentEquity > state.peakEquity) state.peakEquity = state.currentEquity;
    state.maxDrawdown = state.peakEquity > 0
      ? (state.peakEquity - Math.min(state.peakEquity, state.currentEquity)) / state.peakEquity
      : 0;
    state.equityCurve.push({
      ts: closed.closedAt!,
      equity: state.currentEquity,
      cumulativePnL: state.realizedPnL,
    });
    if (state.equityCurve.length > MAX_EQUITY_CURVE_POINTS) {
      state.equityCurve = state.equityCurve.slice(-MAX_EQUITY_CURVE_POINTS);
    }
    state.lastUpdate = new Date().toISOString();
  } catch {
    // non-fatal
  }
}

export function updateShadowUnrealized(profileId: string, unrealized: number): void {
  const state = profileStates.get(profileId);
  if (!state) return;
  state.unrealizedPnL = unrealized;
  state.currentEquity = state.startingCapital + state.realizedPnL + unrealized;
  state.availableCapital = Math.max(0, state.currentEquity - state.reservedCapital);
}

export function recordRejection(profileId: string, reason: string): void {
  let counts = rejectionCounts.get(profileId);
  if (!counts) {
    counts = new Map();
    rejectionCounts.set(profileId, counts);
  }
  counts.set(reason, (counts.get(reason) ?? 0) + 1);
}

/** Per-profile rejection counts for audit instrumentation. No business logic. */
export function getRejectionCountsByProfile(): Record<string, Record<string, number>> {
  const out: Record<string, Record<string, number>> = {};
  Array.from(rejectionCounts.entries()).forEach(([profileId, counts]) => {
    out[profileId] = Object.fromEntries(Array.from(counts.entries()));
  });
  return out;
}

export function getShadowAnalytics(profileId: string): ShadowAnalytics | null {
  const state = profileStates.get(profileId);
  if (!state) return null;

  const closed = state.closedTrades.filter((t) => t.status === "closed");
  const rejectionTotal = Array.from(rejectionCounts.get(profileId)?.values() ?? []).reduce((a, b) => a + b, 0);
  const totalAttempts = closed.length + rejectionTotal || 1;
  const rejections = Array.from(rejectionCounts.get(profileId)?.entries() ?? []);
  const topReasons: Record<string, number> = {};
  rejections.sort((a, b) => b[1] - a[1]).slice(0, 5).forEach(([r, c]) => { topReasons[r] = c; });

  let winRate = 0;
  let avgPnL = 0;
  let avgReturn = 0;
  let avgHolding = 0;
  let avgObserved = 0;
  let avgCapturable = 0;
  let avgFilled = 0;
  const pnlByType: Record<string, number> = {};
  const pnlBySource: Record<string, number> = {};
  let totalRequested = 0;

  if (closed.length > 0) {
    const wins = closed.filter((t) => (t.realizedPnL ?? 0) > 0).length;
    winRate = wins / closed.length;
    avgPnL = closed.reduce((s, t) => s + (t.realizedPnL ?? 0), 0) / closed.length;
    avgReturn = closed.reduce((s, t) => s + (t.realizedReturn ?? 0), 0) / closed.length;
    avgHolding = closed.reduce((s, t) => s + (t.holdingTimeMs ?? 0), 0) / closed.length;
    avgObserved = closed.reduce((s, t) => s + (t.observedEdgeAtEntry ?? 0), 0) / closed.length;
    avgCapturable = closed.reduce((s, t) => s + (t.capturableEdgeAtEntry ?? 0), 0) / closed.length;
    avgFilled = closed.reduce((s, t) => s + (t.filledCapital ?? 0), 0) / closed.length;
    closed.forEach((t) => {
      const p = t.realizedPnL ?? 0;
      const type = t.opportunityType ?? "unknown";
      pnlByType[type] = (pnlByType[type] ?? 0) + p;
      const src = t.sourceType ?? "standard";
      pnlBySource[src] = (pnlBySource[src] ?? 0) + p;
    });
  }

  const totalPnL = state.realizedPnL;
  const utilization = state.startingCapital > 0 ? state.reservedCapital / state.startingCapital : 0;
  const rejectionRate = totalAttempts > 0 ? rejectionTotal / totalAttempts : 0;
  const edgeCaptureRatio = avgObserved > 0 ? avgCapturable / avgObserved : 0;
  const fillEfficiency = avgFilled > 0 ? Math.min(1, avgFilled / Math.max(1, avgFilled * 1.2)) : 0;
  const executionPenalty = avgObserved - avgCapturable;

  return {
    totalTrades: state.activeTrades.length + closed.length,
    closedTrades: closed.length,
    activeTrades: state.activeTrades.length,
    winRate,
    avgPnL,
    avgReturn,
    totalPnL,
    currentEquity: state.currentEquity,
    maxDrawdown: state.maxDrawdown,
    avgHoldingTimeMs: Math.round(avgHolding),
    avgObservedEdgeAtEntry: avgObserved,
    avgCapturableEdgeAtEntry: avgCapturable,
    avgFilledCapital: avgFilled,
    capacityUtilizationRate: utilization,
    rejectionRate,
    topRejectionReasons: topReasons,
    pnlByOpportunityType: pnlByType,
    pnlBySourceType: pnlBySource,
    edgeCaptureRatio,
    fillEfficiency,
    executionPenaltyEstimate: executionPenalty,
  };
}

export function getProfileExposure(profileId: string): {
  exposureByCluster: Record<string, number>;
  exposureByMarket: Record<string, number>;
} {
  const state = profileStates.get(profileId);
  if (!state) return { exposureByCluster: {}, exposureByMarket: {} };
  return {
    exposureByCluster: { ...state.exposureByCluster },
    exposureByMarket: { ...state.exposureByMarket },
  };
}

export function updateProfileExposure(
  profileId: string,
  exposureByCluster: Record<string, number>,
  exposureByMarket: Record<string, number>
): void {
  const state = profileStates.get(profileId);
  if (!state) return;
  state.exposureByCluster = exposureByCluster;
  state.exposureByMarket = exposureByMarket;
}
