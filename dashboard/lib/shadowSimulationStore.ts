/**
 * Shadow Simulation Store — per-profile portfolio state, trades, analytics.
 * Bounded in memory; isolated per profile.
 * Persists closed trades to durable storage for rehydration after redeploys.
 */

import type { PaperSourceType, PaperOpportunityType } from "./paperTypes";
import type { ShadowProfileConfig } from "./shadowSimulationProfiles";
import {
  persistClosedTrades,
  restoreClosedTrades,
  isPersistenceAvailable,
  getPersistencePath,
} from "./shadowClosedTradePersistence";
import { getProfileById, SHADOW_PROFILES } from "./shadowSimulationProfiles";

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

  /** Narrow challenger: persistido no open para audit filtrar apenas narrowFilterMatchAtOpen=true */
  narrowTargetPairKeyAtOpen?: string | null;
  narrowTargetEdgeBucketAtOpen?: string | null;
  narrowTargetFillBucketAtOpen?: string | null;
  narrowFilterMatchAtOpen?: boolean;
  narrowTargetVersion?: string | null;

  /** Entry challenger counterfactual: baseline abriu mas challenger filtrou no mesmo ciclo */
  capfloorFilteredSameCycle?: boolean;
  degratioFilteredSameCycle?: boolean;
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

let rehydratedAt: string | null = null;
let rehydratedClosedTradesCount = 0;
const rehydratedTradeIds = new Set<string>();

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

export interface PersistenceStatus {
  persistenceMode: "file";
  persistencePath: string;
  /** True only if the persistence file exists at the resolved path */
  persistenceFileExists: boolean;
  /** True only if persisted history was actually loaded and count > 0 */
  persistedHistoryAvailable: boolean;
  /** Number of closed trades loaded from persistence during rehydration */
  persistedClosedTradesCount: number;
  /** Number of closed trades currently in memory across all profiles */
  inMemoryClosedTradesCount: number;
  /** Timestamp when rehydration last ran */
  rehydratedAt: string | null;
}

export function getPersistenceStatus(): PersistenceStatus {
  const fileExists = isPersistenceAvailable();
  const inMemoryCount = Array.from(profileStates.values()).reduce(
    (s, state) =>
      s +
      state.closedTrades.filter((t) => t.status === "closed" && t.closedAt)
        .length,
    0
  );
  return {
    persistenceMode: "file",
    persistencePath: getPersistencePath(),
    persistenceFileExists: fileExists,
    persistedHistoryAvailable: rehydratedClosedTradesCount > 0,
    persistedClosedTradesCount: rehydratedClosedTradesCount,
    inMemoryClosedTradesCount: inMemoryCount,
    rehydratedAt,
  };
}

/**
 * Load persisted closed trades into in-memory store. Called once at startup.
 * Only restores for profiles with known config (SHADOW_PROFILES).
 */
export function rehydrateFromPersistence(): void {
  if (rehydratedAt) return;

  const snapshot = restoreClosedTrades();
  if (!snapshot?.byProfile || Object.keys(snapshot.byProfile).length === 0) {
    rehydratedAt = new Date().toISOString();
    return;
  }

  let totalRestored = 0;
  for (const config of SHADOW_PROFILES) {
    const trades = snapshot.byProfile[config.profileId];
    if (!trades?.length) continue;

    const valid = trades.filter(
      (t): t is ShadowTrade => !!(t?.tradeId && t?.status === "closed" && t?.closedAt)
    );
    if (valid.length === 0) continue;

    const sorted = [...valid].sort(
      (a, b) =>
        new Date(a.closedAt!).getTime() - new Date(b.closedAt!).getTime()
    );
    const deduped = sorted.filter(
      (t, i, arr) => i === 0 || arr[i - 1].tradeId !== t.tradeId
    );
    const toRestore = deduped.slice(-MAX_CLOSED_PER_PROFILE);

    const state = ensureProfileState(config);
    state.closedTrades = toRestore;
    for (const t of toRestore) rehydratedTradeIds.add(t.tradeId);
    state.realizedPnL = toRestore.reduce((s, t) => s + (t.realizedPnL ?? 0), 0);
    state.currentEquity = state.startingCapital + state.realizedPnL + state.unrealizedPnL;
    state.availableCapital = Math.max(0, state.currentEquity - state.reservedCapital);

    let peak = state.startingCapital;
    const curve: Array<{ ts: string; equity: number; cumulativePnL: number }> = [
      { ts: new Date().toISOString(), equity: state.startingCapital, cumulativePnL: 0 },
    ];
    let cum = 0;
    for (const t of toRestore) {
      cum += t.realizedPnL ?? 0;
      const equity = state.startingCapital + cum;
      peak = Math.max(peak, equity);
      curve.push({
        ts: t.closedAt!,
        equity,
        cumulativePnL: cum,
      });
    }
    state.peakEquity = peak;
    state.maxDrawdown =
      peak > 0 ? (peak - Math.min(peak, state.currentEquity)) / peak : 0;
    state.equityCurve =
      curve.length > MAX_EQUITY_CURVE_POINTS
        ? curve.slice(-MAX_EQUITY_CURVE_POINTS)
        : curve;
    state.lastUpdate = new Date().toISOString();

    totalRestored += toRestore.length;
  }

  rehydratedAt = new Date().toISOString();
  rehydratedClosedTradesCount = totalRestored;

  if (totalRestored > 0) {
    console.log(
      `[ShadowStore] Rehydrated ${totalRestored} closed trades into ${Object.keys(snapshot.byProfile).length} profiles`
    );
  }
}

/** IDs de trades restaurados na reidratação. Usado para separar histórico vs fluxo novo no audit. */
export function getRehydratedTradeIds(): ReadonlySet<string> {
  return rehydratedTradeIds;
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

    const byProfile: Record<string, ShadowTrade[]> = {};
    for (const [, s] of Array.from(profileStates.entries())) {
      const closed = s.closedTrades.filter(
        (t: ShadowTrade) => t.status === "closed" && t.closedAt
      );
      if (closed.length > 0) byProfile[s.profileId] = closed;
    }
    persistClosedTrades(byProfile);
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

/** Anota trade baseline como "challenger filtrou no mesmo ciclo" para counterfactual no audit */
export function annotateBaselineTradeChallengerRejection(
  baselineProfileId: string,
  opportunityId: string,
  field: "capfloorFilteredSameCycle" | "degratioFilteredSameCycle"
): void {
  const state = profileStates.get(baselineProfileId);
  if (!state) return;
  const t = state.activeTrades.find((x) => x.opportunityId === opportunityId);
  if (t) {
    if (field === "capfloorFilteredSameCycle") t.capfloorFilteredSameCycle = true;
    else t.degratioFilteredSameCycle = true;
  }
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
