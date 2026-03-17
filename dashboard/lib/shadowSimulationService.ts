/**
 * Shadow Simulation Service — runs realistic shadow simulation in background.
 * Combines latency, decay, impact; per-profile state; starts automatically.
 */

import { getGraphOpportunities, ensureGraphScanning } from "./graphScanService";
import { ensureRunning as ensureMarketDataRunning } from "./marketDataService";
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
import { computeClosedTradeAudit } from "./shadowClosedTradeAudit";
import {
  computeAdaptiveCalibration,
  buildSyntheticChallengerSpecForActivation,
} from "./adaptiveCalibrationEngine";
import {
  ensureProfileState,
  addShadowTrade,
  closeShadowTrade,
  updateShadowUnrealized,
  recordRejection,
  getShadowProfileState,
  getProfileExposure,
  getAllShadowProfiles,
  rehydrateFromPersistence,
  type ShadowTrade,
} from "./shadowSimulationStore";
import { logTradeRejection, type TradeRejectionReason } from "./tradeRejectionLogger";
import {
  incrementEvaluateCall,
  incrementExecutionCall,
  incrementShadowTradeOpened,
  incrementEarlyExit,
} from "./shadowPipelineDiagnostics";
import {
  recordEntryDecision,
  SELECTION_CYCLE_BUCKET_MS,
} from "./shadowSelectionDiagnostics";
import {
  recordEvaluated,
  recordRejectedByEntryThreshold,
  recordRejectedByFillGuard,
  recordRejectedByOther,
  recordReachedFillGuardDecision,
} from "./fillGuardDiagnostics";
import { recordThresholdDecision } from "./entryThresholdCausalDiagnostics";
import {
  recordCycleCompleted,
  recordCycleOpportunityIteration,
  recordOpportunityReachedProfile,
  recordReachedSimulateRealisticEntry,
  recordReachedThresholdCheck,
  recordSkippedBeforeThreshold,
  recordEvaluateOpportunityCalled,
} from "./entryThresholdFlowDiagnostics";
import {
  recordShadowBootAttempted,
  recordShadowBootCompleted,
  recordShadowBootFailed,
  recordShadowLoopStarted,
  recordShadowLoopCompleted,
  recordSchedulerScheduled,
} from "./shadowRuntimeDiagnostics";
import { recordStandardFetch, recordGraphFetch, recordMerged } from "./marketSourceDiagnostics";
import {
  isPairMatch,
  isEdgeBucketMatch,
  isFillBucketMatch,
} from "./narrowChallengerHelpers";
import {
  recordOpportunitiesSeen,
  recordRejectedByPairMismatch,
  recordRejectedByEdgeBucketMismatch,
  recordRejectedByFillBucketMismatch,
  recordPassedAllNarrowFilters,
} from "./narrowChallengerDiagnostics";
import type { NormalizedPaperOpportunity } from "./paperTypes";
import type { PersistenceData } from "./edgeDecayModel";

const CYCLE_INTERVAL_MS = 10_000;
const VERBOSE = process.env.WORKER_VERBOSE_LOGS === "1";
/** 0 = next tick; evita shadowLoopStarted=false no primeiro audit por delay de 6s */
const INITIAL_DELAY_MS = 0;

/** Shadow-only: refuse to open trades with dust fills; audit showed avgFilledCapital e-12, 100% loss rate. */
const MIN_FILLED_CAPITAL_USD = 0.5;

/** Registry of enabled adaptive challenger configs — populated by getProfilesForExecution */
const challengerConfigRegistry = new Map<string, ShadowProfileConfig>();

/**
 * Parse ENABLED_ADAPTIVE_CHALLENGERS env (comma-separated profileIds).
 * Only explicitly listed challenger profileIds may execute.
 */
function parseEnabledAdaptiveChallengers(): Set<string> {
  const raw = process.env.ENABLED_ADAPTIVE_CHALLENGERS ?? "";
  return new Set(raw.split(",").map((s) => s.trim()).filter(Boolean));
}

/** Pair for comparison: when either is enabled, run both so audit can compare challenger vs winner. */
const EXITREFINE_V1 = "shadow_1000_adapt_captrade_exitrefine_v1";
const ENTRYCAL_BIND_V1 = "shadow_1000_adapt_captrade_exitrefine_entrycal_bind_v1";
const FILLGUARD_CAL_V1 = "shadow_1000_adapt_captrade_exitrefine_fillguard_cal_v1";
const ENTRYCAL_V1 = "shadow_1000_adapt_captrade_exitrefine_entrycal_v1";

/**
 * Returns baseline profiles + explicitly enabled adaptive challengers.
 * Challengers are only included when they have a current spec from the adaptive layer.
 * When either exitrefine_v1 or entrycal_bind_v1 is enabled, both run for comparison.
 * fillguard_cal_v1 explicitly removed from active comparison — replaced by entrycal_bind (0.025).
 * When entrycal_v1 was enabled (pre-switch), run exitrefine+entrycal_bind instead (entrycal desativado).
 */
export function getProfilesForExecution(): ShadowProfileConfig[] {
  const base = getEnabledProfiles();
  const profiles = getAllShadowProfiles();
  const audit = computeClosedTradeAudit(profiles, getProfileConfig);
  const adaptive = computeAdaptiveCalibration(audit);
  const enabledIds = parseEnabledAdaptiveChallengers();
  const effectiveIds = new Set(enabledIds);
  if (enabledIds.has(EXITREFINE_V1) || enabledIds.has(ENTRYCAL_BIND_V1)) {
    effectiveIds.add(EXITREFINE_V1);
    effectiveIds.add(ENTRYCAL_BIND_V1);
    effectiveIds.delete(FILLGUARD_CAL_V1); // no longer in active comparison
  } else if (enabledIds.has(ENTRYCAL_V1)) {
    effectiveIds.delete(ENTRYCAL_V1);
    effectiveIds.add(EXITREFINE_V1);
    effectiveIds.add(ENTRYCAL_BIND_V1);
  }
  const challengerConfigs = adaptive.adaptiveChallengers
    .filter((c) => effectiveIds.has(c.profileId))
    .map((c) => c.fullConfig);

  challengerConfigRegistry.clear();
  const allChallengerConfigs: ShadowProfileConfig[] = [];
  for (const cfg of challengerConfigs) {
    challengerConfigRegistry.set(cfg.profileId, cfg);
    allChallengerConfigs.push(cfg);
  }
  for (const cfg of challengerConfigs) {
    ensureProfileState(cfg);
  }

  // Materialization fix: ensure enabled challengers that inherit from another adaptive
  // challenger (baseProfileId = shadow_1000_adapt_captrade_v1) get state when the base
  // has no state. Without this, entryfloor would not appear in getAllShadowProfiles().
  // Also: challengers explicitly enabled but NOT in adaptiveChallengers (e.g. recommendation
  // conditions not met) get a synthetic spec so activation works regardless.
  const materializedIds = new Set(getAllShadowProfiles().map((p) => p.profileId));
  for (const id of Array.from(effectiveIds)) {
    if (!materializedIds.has(id)) {
      const spec =
        adaptive.adaptiveChallengers.find((c) => c.profileId === id) ??
        buildSyntheticChallengerSpecForActivation(id);
      if (spec?.fullConfig) {
        challengerConfigRegistry.set(id, spec.fullConfig);
        ensureProfileState(spec.fullConfig);
        allChallengerConfigs.push(spec.fullConfig);
      }
    }
  }

  return [...base, ...allChallengerConfigs];
}

/**
 * Get profile config by profileId — includes baseline profiles and enabled challengers.
 */
export function getProfileConfig(profileId: string): ShadowProfileConfig | undefined {
  return getProfileById(profileId) ?? challengerConfigRegistry.get(profileId);
}

let loopStarted = false;
let lastUpdateMs = 0;
let lastCycleOk = true;
let opportunitiesSeenLastCycle = 0;

/** Normalized pair key for cross-market identification (sorted market ids joined with '+') */
function makePairKey(marketsInvolved: Array<{ marketId: string; question?: string }>): string {
  if (!marketsInvolved?.length) return "";
  const ids = marketsInvolved.map((m) => m.marketId).filter(Boolean);
  if (ids.length === 0) return "";
  return [...ids].sort().join("+");
}

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
    const port = process.env.PORT || "3000";
    const base = typeof window !== "undefined" ? "" : `http://127.0.0.1:${port}`;
    const res = await fetch(`${base}/api/opportunities`, {
      cache: "no-store",
      signal: AbortSignal.timeout(5000),
    });
    if (!res.ok) {
      recordStandardFetch(0, `HTTP ${res.status}`);
      return [];
    }
    const data = await res.json();
    const opps = data.opportunities || [];
    const result = opps.map((o: Record<string, unknown>) => normalizeStandard(o));
    recordStandardFetch(result.length);
    return result;
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    recordStandardFetch(0, msg);
    return [];
  }
}

function fetchGraphOpportunities(): NormalizedPaperOpportunity[] {
  try {
    const ranked = getGraphOpportunities();
    const result = ranked.map((o) => normalizeGraph(o));
    recordGraphFetch(result.length);
    return result;
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    recordGraphFetch(0, msg);
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
  recordShadowLoopStarted();
  const t0 = Date.now();
  const cycleBucket = Math.floor(t0 / SELECTION_CYCLE_BUCKET_MS);
  Promise.all([fetchStandardOpportunities(), Promise.resolve(fetchGraphOpportunities())])
    .then(([stdOpps, graphOpps]) => {
      const merged = [...graphOpps, ...stdOpps];
      recordMerged(merged.length);
      opportunitiesSeenLastCycle = merged.length;
      const profiles = getProfilesForExecution();
      recordCycleCompleted(merged.length, profiles.length);
      const persistenceData = getPersistenceData();
      const capacityResults = estimateBatchCapacity(merged);
      const oppMap = new Map(merged.map((o, i) => [o.opportunityId, { opp: o, capacity: capacityResults[i] }]));

      for (const profile of profiles) {
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
            recordCycleOpportunityIteration(profile.profileId);
            if (capacity.recommendedCapital <= 0) {
              recordSkippedBeforeThreshold(profile.profileId, "capacity_zero");
              continue;
            }
            if (opp.confidence < profile.minConfidenceToTrade) {
              recordSkippedBeforeThreshold(profile.profileId, "confidence_below_threshold");
              continue;
            }
            if (activeOppIds.has(opp.opportunityId)) {
              recordSkippedBeforeThreshold(profile.profileId, "trade_already_active");
              continue;
            }
            const pairKey = makePairKey(opp.marketsInvolved ?? []);
            if (
              pairKey &&
              profile.excludedPairKeys?.length &&
              profile.excludedPairKeys.includes(pairKey)
            ) {
              recordSkippedBeforeThreshold(profile.profileId, "pair_excluded");
              continue;
            }
            recordOpportunityReachedProfile(profile.profileId);
            recordReachedSimulateRealisticEntry(profile.profileId);

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

            recordEvaluated(profile.profileId);
            const minEdge = profile.minCapturableEdgeToTrade ?? profile.minNetCapturableEdgeToTrade;
            const entryResult = simulateRealisticEntry(
              opp,
              capacity,
              freshProfileState,
              persistenceData,
              {
                latencyProfile: profile.latencyProfile,
                impactConfig: { impactAlpha: profile.impactAlpha },
                minCapturableEdgeToTrade: minEdge,
                feeBuffer: profile.feeBuffer,
                liquidityHaircut: profile.liquidityHaircut,
              }
            );

            const thresholdPassed = entryResult.rejectionReason !== "net_edge_below_threshold";
            const metricCompared = entryResult.netEdgeAfterImpact ?? entryResult.capturableEdgeBeforeImpact ?? 0;
            recordThresholdDecision(
              profile.profileId,
              opp.opportunityId,
              pairKey ?? "",
              metricCompared,
              minEdge,
              thresholdPassed ? "accepted" : "rejected",
              thresholdPassed ? undefined : entryResult.rejectionReason,
              cycleBucket
            );
            recordReachedThresholdCheck(profile.profileId);

            if (entryResult.rejectionReason) {
              if (entryResult.rejectionReason === "net_edge_below_threshold") {
                recordRejectedByEntryThreshold(profile.profileId);
              } else {
                recordRejectedByOther(profile.profileId);
              }
              recordRejection(profile.profileId, entryResult.rejectionReason);
              recordEntryDecision(
                profile.profileId,
                opp.opportunityId,
                cycleBucket,
                false,
                entryResult.rejectionReason,
                entryResult.capturableEdgeBeforeImpact
              );
              const reasonMap: Record<string, import("./tradeRejectionLogger").TradeRejectionReason> = {
                insufficient_capital_or_exposure_limit: "TRADE_SIZE_TOO_SMALL",
                net_edge_below_threshold: "EDGE_BELOW_THRESHOLD",
                fill_rejected: "INSUFFICIENT_LIQUIDITY",
              };
              const reason = reasonMap[entryResult.rejectionReason] ?? "UNKNOWN";
              logTradeRejection({
                timestamp: Date.now(),
                marketId: opp.marketsInvolved[0]?.marketId ?? opp.opportunityId,
                edgeBps: Math.round((entryResult.observedEdge ?? opp.edge) * 10000),
                reason,
              });
              continue;
            }
            if (entryResult.filledCapital <= 0) {
              recordRejectedByOther(profile.profileId);
              recordEntryDecision(
                profile.profileId,
                opp.opportunityId,
                cycleBucket,
                false,
                "filled_capital_zero",
                entryResult.capturableEdgeBeforeImpact
              );
              continue;
            }
            if (entryResult.filledCapital < MIN_FILLED_CAPITAL_USD) {
              recordRejectedByOther(profile.profileId);
              recordRejection(profile.profileId, "fill_below_minimum");
              recordEntryDecision(
                profile.profileId,
                opp.opportunityId,
                cycleBucket,
                false,
                "fill_below_minimum",
                entryResult.capturableEdgeBeforeImpact
              );
              continue;
            }
            if (profile.entryPairPenalties && pairKey) {
              const penalty = profile.entryPairPenalties[pairKey] ?? 0;
              const effectiveEdge = entryResult.capturableEdgeBeforeImpact - penalty;
              if (effectiveEdge < minEdge) {
                recordRejectedByOther(profile.profileId);
                recordRejection(profile.profileId, "entry_pair_penalty");
                recordEntryDecision(
                  profile.profileId,
                  opp.opportunityId,
                  cycleBucket,
                  false,
                  "entry_pair_penalty",
                  entryResult.capturableEdgeBeforeImpact
                );
                continue;
              }
            }
            const requestedCapital = entryResult.requestedCapital ?? 0;
            const fillRatio =
              requestedCapital > 0 ? entryResult.filledCapital / requestedCapital : 1;
            if (profile.minFillRatioToTrade != null) {
              recordReachedFillGuardDecision(profile.profileId);
            }
            if (
              profile.minFillRatioToTrade != null &&
              fillRatio < profile.minFillRatioToTrade
            ) {
              recordRejectedByFillGuard(profile.profileId, fillRatio);
              recordRejection(profile.profileId, "fill_ratio_below_threshold");
              recordEntryDecision(
                profile.profileId,
                opp.opportunityId,
                cycleBucket,
                false,
                "fill_ratio_below_threshold",
                entryResult.capturableEdgeBeforeImpact
              );
              continue;
            }

            recordEntryDecision(
              profile.profileId,
              opp.opportunityId,
              cycleBucket,
              true,
              undefined,
              entryResult.capturableEdgeBeforeImpact
            );
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
              requestedCapital: requestedCapital || undefined,
              fillRatio: fillRatio ?? null,
              entryImpactBps:
                entryResult.entrySlippage != null
                  ? Math.round(entryResult.entrySlippage * 10000)
                  : null,
              pairKey: makePairKey(opp.marketsInvolved ?? []) || null,
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

              const edgeAtExit = latestState?.edge;
              const capturableAtEntry = t.capturableEdgeAtEntry ?? 0;
              const entryToExitPriceMove =
                t.effectiveEntryPrice != null && exitResult.effectiveExitPrice != null
                  ? exitResult.effectiveExitPrice - t.effectiveEntryPrice
                  : undefined;

              closeShadowTrade(profile.profileId, t.tradeId, {
                closedAt: exitResult.exitLatencyMs > 0 ? new Date(now + exitResult.exitLatencyMs).toISOString() : new Date().toISOString(),
                effectiveExitPrice: exitResult.effectiveExitPrice,
                realizedPnL: exitResult.realizedPnL,
                realizedReturn: exitResult.realizedReturn,
                holdingTimeMs: holdingMs,
                exitReason: exitResult.exitReason,
                exitImpactBps:
                  exitResult.exitSlippage != null
                    ? Math.round(exitResult.exitSlippage * 10000)
                    : null,
                edgeAtExit: edgeAtExit ?? null,
                edgeDecayDuringHold:
                  edgeAtExit != null && typeof capturableAtEntry === "number"
                    ? edgeAtExit - capturableAtEntry
                    : null,
                entryToExitPriceMove: entryToExitPriceMove ?? null,
                closeContext: {
                  exitReason: exitResult.exitReason,
                  edgeAtExit: edgeAtExit ?? undefined,
                },
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
      recordShadowLoopCompleted();
    })
    .catch((err) => {
      lastCycleOk = false;
      recordStandardFetch(0, err instanceof Error ? err.message : String(err));
      console.warn("[ShadowSim] cycle failed:", err?.message ?? err);
    });
}

/**
 * Evaluate a single opportunity (from scanner) and either open a shadow trade or log rejection.
 * Called by executionDispatcher when opportunities are produced.
 */
export function evaluateOpportunity(opportunity: Record<string, unknown>): void {
  ensureShadowSimulation();
  incrementEvaluateCall();
  recordEvaluateOpportunityCalled();

  const isGraph = Array.isArray(opportunity.marketsInvolved) && (opportunity.marketsInvolved as unknown[]).length > 0;
  const opp: NormalizedPaperOpportunity = isGraph
    ? normalizeGraph(opportunity as Parameters<typeof normalizeGraph>[0])
    : normalizeStandard(opportunity);
  const capacity = estimateOpportunityCapacity(opp);
  const marketId = opp.marketsInvolved[0]?.marketId ?? opp.opportunityId;
  const edgeBps = Math.round((opp.edge ?? 0) * 10000);
  const persistenceData = getPersistenceData();
  const reasonMap: Record<string, TradeRejectionReason> = {
    insufficient_capital_or_exposure_limit: "TRADE_SIZE_TOO_SMALL",
    net_edge_below_threshold: "EDGE_BELOW_THRESHOLD",
    fill_rejected: "INSUFFICIENT_LIQUIDITY",
  };

  const enabledProfiles = getProfilesForExecution();
  if (enabledProfiles.length === 0) {
    incrementEarlyExit("NO_ENABLED_PROFILES");
    if (VERBOSE) console.log("[DIAGNOSTICS] EARLY EXIT", { reason: "NO_ENABLED_PROFILES", marketId });
  }

  for (const profile of enabledProfiles) {
    try {
      recordCycleOpportunityIteration(profile.profileId);
      ensureProfileState(profile);
      const state = getShadowProfileState(profile.profileId);
      const exposure = getProfileExposure(profile.profileId);
      const activeOppIds = new Set((state?.activeTrades ?? []).map((t) => t.opportunityId));
      if (activeOppIds.has(opp.opportunityId)) {
        recordSkippedBeforeThreshold(profile.profileId, "trade_already_active");
        incrementEarlyExit("TRADE_ALREADY_ACTIVE");
        if (VERBOSE) console.log("[DIAGNOSTICS] EARLY EXIT", { reason: "TRADE_ALREADY_ACTIVE", marketId });
        continue;
      }

      if (capacity.recommendedCapital <= 0) {
        recordSkippedBeforeThreshold(profile.profileId, "capacity_zero");
        incrementEarlyExit("RECOMMENDED_CAPITAL_ZERO_OR_NEGATIVE");
        if (VERBOSE) console.log("[DIAGNOSTICS] CAPITAL REJECTION", { marketId, recommendedCapital: capacity.recommendedCapital });
        if (VERBOSE) console.log("[DIAGNOSTICS] EARLY EXIT", { reason: "RECOMMENDED_CAPITAL_ZERO_OR_NEGATIVE", marketId });
        logTradeRejection({ timestamp: Date.now(), marketId, edgeBps, reason: "EDGE_BELOW_THRESHOLD" });
        continue;
      }
      if (opp.confidence < profile.minConfidenceToTrade) {
        recordSkippedBeforeThreshold(profile.profileId, "confidence_below_threshold");
        incrementEarlyExit("CONFIDENCE_BELOW_THRESHOLD");
        if (VERBOSE) console.log("[DIAGNOSTICS] CONFIDENCE REJECTION", {
          marketId,
          confidence: opp.confidence,
          minConfidenceToTrade: profile.minConfidenceToTrade,
        });
        if (VERBOSE) console.log("[DIAGNOSTICS] EARLY EXIT", { reason: "CONFIDENCE_BELOW_THRESHOLD", marketId });
        logTradeRejection({ timestamp: Date.now(), marketId, edgeBps, reason: "LATENCY_RISK" });
        continue;
      }
      const pairKey = makePairKey(opp.marketsInvolved ?? []);
      if (
        pairKey &&
        profile.excludedPairKeys?.length &&
        profile.excludedPairKeys.includes(pairKey)
      ) {
        recordSkippedBeforeThreshold(profile.profileId, "pair_excluded");
        incrementEarlyExit("PAIR_EXCLUDED_BY_CHALLENGER");
        continue;
      }
      recordOpportunityReachedProfile(profile.profileId);
      recordReachedSimulateRealisticEntry(profile.profileId);
      const freshExposure = getProfileExposure(profile.profileId);
      const freshState = getShadowProfileState(profile.profileId);
      const profileState = {
        availableCapital: freshState?.availableCapital ?? profile.startingCapital,
        maxCapitalPerTrade: profile.maxCapitalPerTrade,
        maxCapitalPerCluster: profile.maxCapitalPerCluster,
        maxCapitalPerMarket: profile.maxCapitalPerMarket,
        exposureByCluster: freshExposure.exposureByCluster,
        exposureByMarket: freshExposure.exposureByMarket,
      };
      if (VERBOSE && freshState && profileState.availableCapital < 5) {
        const freeRatio = freshState.startingCapital > 0
          ? profileState.availableCapital / freshState.startingCapital
          : 0;
        const otherCaps = [
          capacity.recommendedCapital,
          profile.maxCapitalPerTrade,
          opp.liquidity * 0.1,
          opp.liquidity * 0.08,
        ];
        const bindingCap = profileState.availableCapital <= Math.min(...otherCaps)
          ? "availableCapital"
          : "other";
        console.log("[DIAGNOSTICS] AVAILABLE CAPITAL ANALYSIS", {
          profileId: profile.profileId,
          portfolioEquity: freshState.currentEquity,
          reservedCapital: freshState.reservedCapital,
          availableCapital: profileState.availableCapital,
          activeTrades: freshState.activeTrades.length,
          startingCapital: freshState.startingCapital,
          freeCapitalRatio: freeRatio,
          bindingCap,
          capCapacity: capacity.recommendedCapital,
          capMaxTrade: profile.maxCapitalPerTrade,
          capLiq8Pct: opp.liquidity * 0.08,
        });
      }
      incrementExecutionCall();
      const minEdge = profile.minCapturableEdgeToTrade ?? profile.minNetCapturableEdgeToTrade;
      const entryResult = simulateRealisticEntry(opp, capacity, profileState, persistenceData, {
        latencyProfile: profile.latencyProfile,
        impactConfig: { impactAlpha: profile.impactAlpha },
        minCapturableEdgeToTrade: minEdge,
        feeBuffer: profile.feeBuffer,
        liquidityHaircut: profile.liquidityHaircut,
      });
      if (VERBOSE) console.log("EXECUTION ENGINE RESULT", entryResult);

      const evalCycleBucket = Math.floor(Date.now() / SELECTION_CYCLE_BUCKET_MS) * SELECTION_CYCLE_BUCKET_MS;
      const thresholdPassedEval = entryResult.rejectionReason !== "net_edge_below_threshold";
      const metricComparedEval = entryResult.netEdgeAfterImpact ?? entryResult.capturableEdgeBeforeImpact ?? 0;
      recordThresholdDecision(
        profile.profileId,
        opp.opportunityId,
        pairKey ?? "",
        metricComparedEval,
        minEdge,
        thresholdPassedEval ? "accepted" : "rejected",
        thresholdPassedEval ? undefined : entryResult.rejectionReason,
        evalCycleBucket
      );
      recordReachedThresholdCheck(profile.profileId);

      if (entryResult.rejectionReason) {
        const engineReason =
          entryResult.rejectionReason === "insufficient_capital_or_exposure_limit"
            ? "INSUFFICIENT_CAPITAL_OR_EXPOSURE_LIMIT"
            : entryResult.rejectionReason === "net_edge_below_threshold"
              ? "NET_EDGE_BELOW_THRESHOLD"
              : "FILL_REJECTED";
        incrementEarlyExit(engineReason);
        if (VERBOSE) console.log("[DIAGNOSTICS] FILL REJECTION", {
          marketId,
          filledCapital: entryResult.filledCapital,
          reason: entryResult.rejectionReason,
        });
        if (VERBOSE) console.log("[DIAGNOSTICS] EARLY EXIT", { reason: engineReason, marketId });
        const reason = reasonMap[entryResult.rejectionReason] ?? "UNKNOWN";
        recordRejection(profile.profileId, entryResult.rejectionReason);
        logTradeRejection({
          timestamp: Date.now(),
          marketId,
          edgeBps: Math.round((entryResult.observedEdge ?? opp.edge) * 10000),
          reason,
        });
        continue;
      }
      if (entryResult.filledCapital <= 0) {
        incrementEarlyExit("FILLED_CAPITAL_ZERO");
        if (VERBOSE) console.log("[DIAGNOSTICS] FILL REJECTION", {
          marketId,
          filledCapital: entryResult.filledCapital,
          reason: "filled_capital_zero",
        });
        if (VERBOSE) console.log("[DIAGNOSTICS] EARLY EXIT", { reason: "FILLED_CAPITAL_ZERO", marketId });
        continue;
      }
      if (entryResult.filledCapital < MIN_FILLED_CAPITAL_USD) {
        recordRejection(profile.profileId, "fill_below_minimum");
        continue;
      }
      if (profile.entryPairPenalties && pairKey) {
        const penalty = profile.entryPairPenalties[pairKey] ?? 0;
        const effectiveEdge = entryResult.capturableEdgeBeforeImpact - penalty;
        if (effectiveEdge < minEdge) {
          recordRejection(profile.profileId, "entry_pair_penalty");
          continue;
        }
      }
      const requestedCapital = entryResult.requestedCapital ?? 0;
      const fillRatio = requestedCapital > 0 ? entryResult.filledCapital / requestedCapital : 1;
      if (
        profile.minFillRatioToTrade != null &&
        fillRatio < profile.minFillRatioToTrade
      ) {
        recordRejection(profile.profileId, "fill_ratio_below_threshold");
        continue;
      }
      // Narrow challenger: só abre quando pair/edge/fill batem exatamente (auditável)
      const target = profile.narrowChallengerTarget;
      if (target) {
        recordOpportunitiesSeen(profile.profileId);
        if (!isPairMatch(pairKey ?? null, target.pairKey)) {
          recordRejection(profile.profileId, "narrow_pair_mismatch");
          recordRejectedByPairMismatch(profile.profileId);
          continue;
        }
        if (!isEdgeBucketMatch(entryResult.capturableEdgeBeforeImpact, target.capturableEdgeBucket)) {
          recordRejection(profile.profileId, "narrow_edge_bucket_mismatch");
          recordRejectedByEdgeBucketMismatch(profile.profileId);
          continue;
        }
        if (!isFillBucketMatch(fillRatio, target.fillRatioBucket)) {
          recordRejection(profile.profileId, "narrow_fill_bucket_mismatch");
          recordRejectedByFillBucketMismatch(profile.profileId);
          continue;
        }
        recordPassedAllNarrowFilters(profile.profileId);
      }
      const trade: ShadowTrade = {
        tradeId: `sst-${profile.profileId}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
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
        requestedCapital: requestedCapital || undefined,
        fillRatio: fillRatio ?? null,
        entryImpactBps:
          entryResult.entrySlippage != null
            ? Math.round(entryResult.entrySlippage * 10000)
            : null,
        pairKey: makePairKey(opp.marketsInvolved ?? []) || null,
        realizedPnL: 0,
        realizedReturn: 0,
        holdingTimeMs: 0,
      };
      addShadowTrade(profile.profileId, trade, profile);
      incrementShadowTradeOpened();
      if (VERBOSE) console.log("[DIAGNOSTICS] SHADOW TRADE OPENED", { marketId });
      if (VERBOSE) console.log("SHADOW TRADE EXECUTED", { tradeId: trade.tradeId, filledCapital: trade.filledCapital });
      const updatedState = getShadowProfileState(profile.profileId);
      if (VERBOSE) console.log("SHADOW PORTFOLIO UPDATED", {
        activeTrades: updatedState?.activeTrades?.length ?? 0,
        closedTrades: updatedState?.closedTrades?.length ?? 0,
        equity: updatedState?.currentEquity ?? 0,
      });
    } catch (err) {
      incrementEarlyExit("EVALUATION_CATCH_ERROR");
      if (VERBOSE) console.log("[DIAGNOSTICS] EARLY EXIT", { reason: "EVALUATION_CATCH_ERROR", marketId });
      console.warn(`[ShadowSim] evaluateOpportunity failed for ${opp.opportunityId}:`, err instanceof Error ? err.message : err);
    }
  }
}

export function ensureShadowSimulation(): void {
  recordShadowBootAttempted();
  if (loopStarted) {
    recordShadowBootCompleted();
    return;
  }
  loopStarted = true;
  try {
    rehydrateFromPersistence();
    ensureMarketDataRunning();
    ensureGraphScanning();
    for (const p of getEnabledProfiles()) {
      ensureProfileState(p);
    }
    recordSchedulerScheduled();
    console.log("[ShadowSim] Background shadow simulation started");
    setTimeout(runCycle, INITIAL_DELAY_MS);
    setInterval(runCycle, CYCLE_INTERVAL_MS);
    recordShadowBootCompleted();
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    recordShadowBootFailed(msg);
    throw err;
  }
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
    profilesRunning: getProfilesForExecution().map((p) => p.profileId),
    opportunitiesSeenLastCycle,
    notes: "Realistic shadow simulation with latency, decay, impact",
  };
}
