/**
 * Shadow Simulation Service — runs realistic shadow simulation in background.
 * Combines latency, decay, impact; per-profile state; starts automatically.
 */

import { getGraphOpportunities } from "./graphScanService";
import { getGraphEpisodeSummary } from "./graphEpisodeStore";
import { estimateOpportunityCapacity, estimateBatchCapacity } from "./capitalCapacityEngine";
import {
  simulateRealisticEntry,
  simulateRealisticExit,
  type ActiveShadowTradeState,
} from "./realisticExecutionEngine";
import {
  getEnabledProfiles,
  getProfileById,
  type ShadowProfileConfig,
} from "./shadowSimulationProfiles";
import {
  ensureProfileState,
  addShadowTrade,
  closeShadowTrade,
  updateShadowUnrealized,
  recordRejection,
  getShadowProfileState,
  getProfileExposure,
  type ShadowTrade,
} from "./shadowSimulationStore";
import type { NormalizedPaperOpportunity } from "./paperTypes";
import type { PersistenceData } from "./edgeDecayModel";

const CYCLE_INTERVAL_MS = 10_000;
const INITIAL_DELAY_MS = 6_000;

let loopStarted = false;
let lastUpdateMs = 0;
let lastCycleOk = true;
let opportunitiesSeenLastCycle = 0;

function estimateSpread(opp: { prices?: number[]; liquidity?: number; confidence?: number }): number {
  if (opp.prices && Array.isArray(opp.prices) && opp.prices.length >= 2) {
    const sorted = [...opp.prices].sort((a, b) => b - a);
    return Math.max(0.01, sorted[0] - sorted[sorted.length - 1]);
  }
  return Math.max(0.01, 0.02 * (1 - (opp.confidence || 0.5)));
}

function normalizeStandard(opp: Record<string, unknown>): NormalizedPaperOpportunity {
  const spread = estimateSpread(opp as { prices?: number[]; confidence?: number });
  return {
    opportunityId: String(opp.marketId ?? opp.id ?? ""),
    sourceType: "standard",
    opportunityType: String(opp.type ?? "overround") as NormalizedPaperOpportunity["opportunityType"],
    marketsInvolved: [{ marketId: String(opp.marketId ?? ""), question: String(opp.question ?? "") }],
    edge: Number(opp.edge ?? 0),
    confidence: Number(opp.confidence ?? 0),
    liquidity: Number(opp.liquidity ?? 0),
    spread,
  };
}

function normalizeGraph(opp: {
  id: string;
  type: string;
  edge: number;
  confidence: number;
  liquidity: number;
  clusterId?: string;
  marketsInvolved: Array<{ marketId: string; question: string }>;
}): NormalizedPaperOpportunity {
  const spread = estimateSpread(opp);
  return {
    opportunityId: opp.id,
    sourceType: "graph",
    opportunityType: opp.type as NormalizedPaperOpportunity["opportunityType"],
    clusterId: opp.clusterId,
    marketsInvolved: opp.marketsInvolved || [],
    edge: opp.edge,
    confidence: opp.confidence,
    liquidity: opp.liquidity,
    spread,
  };
}

async function fetchStandardOpportunities(): Promise<NormalizedPaperOpportunity[]> {
  try {
    const base = typeof window !== "undefined" ? "" : "http://localhost:3000";
    const res = await fetch(`${base}/api/opportunities`, {
      cache: "no-store",
      signal: AbortSignal.timeout(5000),
    });
    if (!res.ok) return [];
    const data = await res.json();
    const opps = data.opportunities || [];
    return opps.map((o: Record<string, unknown>) => normalizeStandard(o));
  } catch {
    return [];
  }
}

function fetchGraphOpportunities(): NormalizedPaperOpportunity[] {
  try {
    const ranked = getGraphOpportunities();
    return ranked.map((o) => normalizeGraph(o));
  } catch {
    return [];
  }
}

function getPersistenceData(): PersistenceData | null {
  try {
    const summary = getGraphEpisodeSummary();
    if (summary.avgActiveDurationMs > 0 || summary.totalEpisodesTracked > 0) {
      return {
        avgActiveDurationMs: summary.avgActiveDurationMs,
        longestRecentEpisodeMs: summary.longestRecentEpisodeMs,
      };
    }
  } catch {
    // non-fatal
  }
  return null;
}

function runCycle(): void {
  const t0 = Date.now();
  Promise.all([fetchStandardOpportunities(), Promise.resolve(fetchGraphOpportunities())])
    .then(([stdOpps, graphOpps]) => {
      const merged = [...graphOpps, ...stdOpps];
      opportunitiesSeenLastCycle = merged.length;
      const persistenceData = getPersistenceData();
      const capacityResults = estimateBatchCapacity(merged);
      const oppMap = new Map(merged.map((o, i) => [o.opportunityId, { opp: o, capacity: capacityResults[i] }]));

      for (const profile of getEnabledProfiles()) {
        try {
          const state = ensureProfileState(profile);
          const exposure = getProfileExposure(profile.profileId);
          const profileState = {
            availableCapital: state.availableCapital,
            maxCapitalPerTrade: profile.maxCapitalPerTrade,
            maxCapitalPerCluster: profile.maxCapitalPerCluster,
            maxCapitalPerMarket: profile.maxCapitalPerMarket,
            exposureByCluster: exposure.exposureByCluster,
            exposureByMarket: exposure.exposureByMarket,
          };

          let opened = 0;
          let closed = 0;

          const activeOppIds = new Set(state.activeTrades.map((t) => t.opportunityId));

          for (const { opp, capacity } of Array.from(oppMap.values())) {
            if (capacity.recommendedCapital <= 0) continue;
            if (opp.confidence < profile.minConfidenceToTrade) continue;
            if (activeOppIds.has(opp.opportunityId)) continue;

            const freshExposure = getProfileExposure(profile.profileId);
            const freshState = getShadowProfileState(profile.profileId);
            const freshProfileState = {
              availableCapital: freshState?.availableCapital ?? profile.startingCapital,
              maxCapitalPerTrade: profile.maxCapitalPerTrade,
              maxCapitalPerCluster: profile.maxCapitalPerCluster,
              maxCapitalPerMarket: profile.maxCapitalPerMarket,
              exposureByCluster: freshExposure.exposureByCluster,
              exposureByMarket: freshExposure.exposureByMarket,
            };

            const entryResult = simulateRealisticEntry(
              opp,
              capacity,
              freshProfileState,
              persistenceData,
              {
                latencyProfile: profile.latencyProfile,
                impactConfig: { impactAlpha: profile.impactAlpha },
                minCapturableEdgeToTrade: profile.minNetCapturableEdgeToTrade,
                feeBuffer: profile.feeBuffer,
                liquidityHaircut: profile.liquidityHaircut,
              }
            );

            if (entryResult.rejectionReason) {
              recordRejection(profile.profileId, entryResult.rejectionReason);
              continue;
            }
            if (entryResult.filledCapital <= 0) continue;

            const tradeId = `sst-${profile.profileId}-${Date.now()}-${opened}`;
            const trade: ShadowTrade = {
              tradeId,
              opportunityId: opp.opportunityId,
              sourceType: opp.sourceType,
              opportunityType: opp.opportunityType,
              clusterId: opp.clusterId,
              marketsInvolved: opp.marketsInvolved,
              openedAt: new Date().toISOString(),
              closedAt: null,
              status: "active",
              observedEdgeAtEntry: entryResult.observedEdge,
              capturableEdgeAtEntry: entryResult.capturableEdgeBeforeImpact,
              effectiveEntryPrice: entryResult.effectiveEntryPrice,
              filledCapital: entryResult.filledCapital,
              realizedPnL: 0,
              realizedReturn: 0,
              holdingTimeMs: 0,
            };

            addShadowTrade(profile.profileId, trade, profile);
            activeOppIds.add(opp.opportunityId);
            opened++;
          }

          const updatedState = getShadowProfileState(profile.profileId);
          if (!updatedState) continue;

          for (const t of [...updatedState.activeTrades]) {
            const latest = oppMap.get(t.opportunityId);
            const latestState = latest ? { edge: latest.opp.edge, confidence: latest.opp.confidence } : null;

            const activeState: ActiveShadowTradeState = {
              tradeId: t.tradeId,
              opportunityId: t.opportunityId,
              openedAt: t.openedAt,
              entryEdge: t.observedEdgeAtEntry,
              capturableEdgeAtEntry: t.capturableEdgeAtEntry,
              effectiveEntryPrice: t.effectiveEntryPrice,
              filledCapital: t.filledCapital,
              maxAdverseExcursion: 0,
              maxFavorableExcursion: 0,
            };

            const now = Date.now();
            const holdingMs = now - new Date(t.openedAt).getTime();
            let shouldClose = false;
            if (holdingMs >= profile.maxHoldingTimeMs) shouldClose = true;
            else if (latestState) {
              const exitPrice = 1 - latestState.edge;
              const pnlPct = (exitPrice - t.effectiveEntryPrice) / Math.max(0.001, t.effectiveEntryPrice);
              if (pnlPct <= -profile.stopLossPct) shouldClose = true;
              else if (pnlPct >= profile.takeProfitPct) shouldClose = true;
              else if (Math.abs(latestState.edge) < 0.005) shouldClose = true;
            } else {
              shouldClose = true;
            }

            if (shouldClose) {
              const exitResult = simulateRealisticExit(activeState, latestState ?? null, {
                latencyProfile: profile.latencyProfile,
                impactConfig: { impactAlpha: profile.impactAlpha },
              });

              closeShadowTrade(profile.profileId, t.tradeId, {
                closedAt: exitResult.exitLatencyMs > 0 ? new Date(now + exitResult.exitLatencyMs).toISOString() : new Date().toISOString(),
                effectiveExitPrice: exitResult.effectiveExitPrice,
                realizedPnL: exitResult.realizedPnL,
                realizedReturn: exitResult.realizedReturn,
                holdingTimeMs: holdingMs,
                exitReason: exitResult.exitReason,
              });
              closed++;
            }
          }

          const finalState = getShadowProfileState(profile.profileId);
          if (finalState) {
            const unrealized = finalState.activeTrades.reduce((s, t) => {
              const exitEst = 1 - t.observedEdgeAtEntry;
              const pnl = t.filledCapital * ((exitEst - t.effectiveEntryPrice) / Math.max(0.001, t.effectiveEntryPrice));
              return s + pnl;
            }, 0);
            updateShadowUnrealized(profile.profileId, unrealized);
          }

          const elapsed = Date.now() - t0;
          const eq = getShadowProfileState(profile.profileId)?.currentEquity ?? profile.startingCapital;
          if (opened > 0 || closed > 0 || merged.length > 0) {
            console.log(
              `[ShadowSim] profile=${profile.profileId} seen=${merged.length} opened=${opened} closed=${closed} equity=${eq.toFixed(2)} duration=${elapsed}ms`
            );
          }
        } catch (err) {
          console.warn(`[ShadowSim] profile ${profile.profileId} failed:`, err instanceof Error ? err.message : err);
        }
      }

      lastUpdateMs = Date.now();
      lastCycleOk = true;
    })
    .catch((err) => {
      lastCycleOk = false;
      console.warn("[ShadowSim] cycle failed:", err?.message ?? err);
    });
}

export function ensureShadowSimulation(): void {
  if (loopStarted) return;
  loopStarted = true;
  for (const p of getEnabledProfiles()) {
    ensureProfileState(p);
  }
  console.log("[ShadowSim] Background shadow simulation started");
  setTimeout(runCycle, INITIAL_DELAY_MS);
  setInterval(runCycle, CYCLE_INTERVAL_MS);
}

export function getShadowSystemStatus(): {
  status: string;
  lastUpdate: string | null;
  profilesRunning: string[];
  opportunitiesSeenLastCycle: number;
  notes: string;
} {
  ensureShadowSimulation();
  return {
    status: lastCycleOk ? "ok" : "degraded",
    lastUpdate: lastUpdateMs > 0 ? new Date(lastUpdateMs).toISOString() : null,
    profilesRunning: getEnabledProfiles().map((p) => p.profileId),
    opportunitiesSeenLastCycle,
    notes: "Realistic shadow simulation with latency, decay, impact",
  };
}
