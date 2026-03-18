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
  updateProfileHeartbeat,
  recordRejection,
  getShadowProfileState,
  getProfileExposure,
  getAllShadowProfiles,
  rehydrateFromPersistence,
  annotateBaselineTradeChallengerRejection,
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
import {
  isStructuralPairMatch,
  isStructuralFillBucketMatch,
  isStructuralEdgeBucketGt5Match,
  getCapturableEdgeBucket,
} from "./structuralChallengerHelpers";
import {
  recordStructuralOpportunitiesSeen,
  recordStructuralRejectedByPairMismatch,
  recordStructuralRejectedByFillBucketMismatch,
  recordStructuralRejectedByEdgeBucketMismatch,
  recordStructuralPassedAllFilters,
} from "./structuralChallengerDiagnostics";
import {
  recordExitKillEvaluated,
  recordExitKillKilled,
} from "./structuralExitKillDiagnostics";
import {
  recordStructuralRiskEvaluated,
  recordStructuralRiskRejectedByPair,
  recordStructuralRiskRejectedByFillBucket,
  recordStructuralRiskRejectedByCapfloor,
  recordStructuralRiskRejectedByDegRatio,
  recordStructuralRiskRejectedByOther,
} from "./structuralRiskManagedDiagnostics";
import {
  recordEntryChallengerDecision,
  BASELINE_PROFILE_ID,
} from "./entryChallengerDiagnostics";
import {
  recordCycleProcessed,
  recordRawOpportunitySeen,
  recordPairEligible,
  recordFillEligible,
  recordCapfloorEligible,
  recordDegratioEligible,
  recordFinalCandidate,
  recordOpenAttempt,
  recordOpened,
} from "./profileEligibilityFunnel";
import type { NormalizedPaperOpportunity } from "./paperTypes";
import type { PersistenceData } from "./edgeDecayModel";

const CYCLE_INTERVAL_MS = 10_000;
const VERBOSE = process.env.WORKER_VERBOSE_LOGS === "1";
/** 0 = next tick; evita shadowLoopStarted=false no primeiro audit por delay de 6s */
const INITIAL_DELAY_MS = 0;

/** Shadow-only: refuse to open trades with dust fills; audit showed avgFilledCapital e-12, 100% loss rate. */
const MIN_FILLED_CAPITAL_USD = 0.5;

/** Early thesis-failure monitoring window for structural risk-managed challenger */
const EARLY_THESIS_MONITORING_WINDOW_MS = 90_000;

/** Per tradeId: consecutive cycles where structural opp was absent */
const structuralRiskConsecutiveAbsentByTradeId = new Map<string, number>();

/** Per tradeId: consecutive cycles where edge <= stagnantFloor (late exit) */
const lateExitStagnantCyclesByTradeId = new Map<string, number>();

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
          recordCycleProcessed(profile.profileId);
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
            recordRawOpportunitySeen(profile.profileId);
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
              const earlyFillRatio =
                (entryResult.requestedCapital ?? 0) > 0
                  ? (entryResult.filledCapital ?? 0) / (entryResult.requestedCapital ?? 1)
                  : undefined;
              recordEntryChallengerDecision(
                profile.profileId,
                opp.opportunityId,
                cycleBucket,
                false,
                entryResult.rejectionReason,
                entryResult.capturableEdgeBeforeImpact,
                entryResult.observedEdge,
                earlyFillRatio,
                pairKey ?? undefined
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
              recordEntryChallengerDecision(
                profile.profileId,
                opp.opportunityId,
                cycleBucket,
                false,
                "filled_capital_zero",
                entryResult.capturableEdgeBeforeImpact,
                entryResult.observedEdge
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
              recordEntryChallengerDecision(
                profile.profileId,
                opp.opportunityId,
                cycleBucket,
                false,
                "fill_below_minimum",
                entryResult.capturableEdgeBeforeImpact,
                entryResult.observedEdge
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
                const preFillRatio =
                  (entryResult.requestedCapital ?? 0) > 0
                    ? (entryResult.filledCapital ?? 0) / (entryResult.requestedCapital ?? 1)
                    : undefined;
                recordEntryChallengerDecision(
                  profile.profileId,
                  opp.opportunityId,
                  cycleBucket,
                  false,
                  "entry_pair_penalty",
                  entryResult.capturableEdgeBeforeImpact,
                  entryResult.observedEdge,
                  preFillRatio,
                  pairKey
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
              recordEntryChallengerDecision(
                profile.profileId,
                opp.opportunityId,
                cycleBucket,
                false,
                "fill_ratio_below_threshold",
                entryResult.capturableEdgeBeforeImpact,
                entryResult.observedEdge,
                fillRatio ?? undefined,
                pairKey ?? undefined
              );
              continue;
            }

            /** Narrow challenger: só abre quando TODOS os filtros narrow forem satisfeitos */
            if (profile.narrowChallengerTarget) {
              recordOpportunitiesSeen(profile.profileId);
              const pairOk = isPairMatch(pairKey ?? null, profile.narrowChallengerTarget.pairKey);
              if (!pairOk) {
                recordRejectedByPairMismatch(profile.profileId);
                recordRejection(profile.profileId, "narrow_pair_mismatch");
                recordEntryDecision(
                  profile.profileId,
                  opp.opportunityId,
                  cycleBucket,
                  false,
                  "narrow_pair_mismatch",
                  entryResult.capturableEdgeBeforeImpact
                );
                continue;
              }
              const edgeOk = isEdgeBucketMatch(
                entryResult.capturableEdgeBeforeImpact ?? 0,
                profile.narrowChallengerTarget.capturableEdgeBucket
              );
              if (!edgeOk) {
                recordRejectedByEdgeBucketMismatch(profile.profileId);
                recordRejection(profile.profileId, "narrow_edge_bucket_mismatch");
                recordEntryDecision(
                  profile.profileId,
                  opp.opportunityId,
                  cycleBucket,
                  false,
                  "narrow_edge_bucket_mismatch",
                  entryResult.capturableEdgeBeforeImpact
                );
                continue;
              }
              const fillOk = isFillBucketMatch(fillRatio, profile.narrowChallengerTarget.fillRatioBucket);
              if (!fillOk) {
                recordRejectedByFillBucketMismatch(profile.profileId);
                recordRejection(profile.profileId, "narrow_fill_bucket_mismatch");
                recordEntryDecision(
                  profile.profileId,
                  opp.opportunityId,
                  cycleBucket,
                  false,
                  "narrow_fill_bucket_mismatch",
                  entryResult.capturableEdgeBeforeImpact
                );
                continue;
              }
              recordPassedAllNarrowFilters(profile.profileId);
            }

            /** Structural challenger: pair set + fill bucket + edge bucket opcional */
            if (profile.structuralChallengerTarget) {
              recordStructuralOpportunitiesSeen(profile.profileId);
              if (!isStructuralPairMatch(pairKey ?? null)) {
                recordStructuralRejectedByPairMismatch(profile.profileId);
                recordRejection(profile.profileId, "structural_pair_mismatch");
                recordEntryDecision(
                  profile.profileId,
                  opp.opportunityId,
                  cycleBucket,
                  false,
                  "structural_pair_mismatch",
                  entryResult.capturableEdgeBeforeImpact
                );
                continue;
              }
              if (!isStructuralFillBucketMatch(fillRatio)) {
                recordStructuralRejectedByFillBucketMismatch(profile.profileId);
                recordRejection(profile.profileId, "structural_fill_bucket_mismatch");
                recordEntryDecision(
                  profile.profileId,
                  opp.opportunityId,
                  cycleBucket,
                  false,
                  "structural_fill_bucket_mismatch",
                  entryResult.capturableEdgeBeforeImpact
                );
                continue;
              }
              if (profile.structuralChallengerTarget.capturableEdgeBucket === ">5%") {
                if (!isStructuralEdgeBucketGt5Match(entryResult.capturableEdgeBeforeImpact ?? 0)) {
                  recordStructuralRejectedByEdgeBucketMismatch(profile.profileId);
                  recordRejection(profile.profileId, "structural_edge_bucket_mismatch");
                  recordEntryDecision(
                    profile.profileId,
                    opp.opportunityId,
                    cycleBucket,
                    false,
                    "structural_edge_bucket_mismatch",
                    entryResult.capturableEdgeBeforeImpact
                  );
                  continue;
                }
              }
              recordStructuralPassedAllFilters(profile.profileId);
            }

            /** Structural risk-managed: pair + fill + capfloor 4.5% + degratio 0.24 */
            const structuralRiskTarget = profile.structuralRiskManagedTarget;
            if (structuralRiskTarget) {
              recordStructuralRiskEvaluated();
              if (!isStructuralPairMatch(pairKey ?? null)) {
                recordStructuralRiskRejectedByPair();
                recordRejection(profile.profileId, "structural_risk_pair_mismatch");
                recordEntryDecision(
                  profile.profileId,
                  opp.opportunityId,
                  cycleBucket,
                  false,
                  "structural_risk_pair_mismatch",
                  entryResult.capturableEdgeBeforeImpact
                );
                continue;
              }
              recordPairEligible(profile.profileId);
              if (!isStructuralFillBucketMatch(fillRatio)) {
                recordStructuralRiskRejectedByFillBucket();
                recordRejection(profile.profileId, "structural_risk_fill_bucket_mismatch");
                recordEntryDecision(
                  profile.profileId,
                  opp.opportunityId,
                  cycleBucket,
                  false,
                  "structural_risk_fill_bucket_mismatch",
                  entryResult.capturableEdgeBeforeImpact
                );
                continue;
              }
              recordFillEligible(profile.profileId);
              const cap = entryResult.capturableEdgeBeforeImpact ?? 0;
              if (cap < structuralRiskTarget.capfloor) {
                recordStructuralRiskRejectedByCapfloor();
                recordRejection(profile.profileId, "structural_risk_capfloor");
                recordEntryDecision(
                  profile.profileId,
                  opp.opportunityId,
                  cycleBucket,
                  false,
                  "structural_risk_capfloor",
                  entryResult.capturableEdgeBeforeImpact
                );
                continue;
              }
              recordCapfloorEligible(profile.profileId);
              const observed = entryResult.observedEdge ?? opp.edge ?? 0;
              const degRatio = cap / Math.max(0.0001, observed);
              if (degRatio < structuralRiskTarget.degRatioMin) {
                recordStructuralRiskRejectedByDegRatio();
                recordRejection(profile.profileId, "structural_risk_degratio");
                recordEntryDecision(
                  profile.profileId,
                  opp.opportunityId,
                  cycleBucket,
                  false,
                  "structural_risk_degratio",
                  entryResult.capturableEdgeBeforeImpact
                );
                continue;
              }
              recordDegratioEligible(profile.profileId);
              if (profile.exitKillTarget) recordExitKillEvaluated();
            }

            /** Entry capfloor challenger: só abre se capturableEdge >= 0.03 */
            if (profile.entryCapfloorMinCapturableEdge != null) {
              const cap = entryResult.capturableEdgeBeforeImpact ?? 0;
              if (cap < profile.entryCapfloorMinCapturableEdge) {
                recordRejection(profile.profileId, "entry_capfloor");
                recordEntryDecision(
                  profile.profileId,
                  opp.opportunityId,
                  cycleBucket,
                  false,
                  "entry_capfloor",
                  entryResult.capturableEdgeBeforeImpact
                );
                const degRatio =
                  (entryResult.capturableEdgeBeforeImpact ?? 0) /
                  Math.max(0.0001, entryResult.observedEdge ?? opp.edge ?? 0);
                recordEntryChallengerDecision(
                  profile.profileId,
                  opp.opportunityId,
                  cycleBucket,
                  false,
                  "entry_capfloor",
                  entryResult.capturableEdgeBeforeImpact,
                  entryResult.observedEdge,
                  fillRatio ?? undefined,
                  pairKey ?? undefined,
                  degRatio
                );
                annotateBaselineTradeChallengerRejection(
                  BASELINE_PROFILE_ID,
                  opp.opportunityId,
                  "capfloorFilteredSameCycle"
                );
                continue;
              }
            }

            /** Entry degratio challenger: só abre se capturable/observed >= 0.22 */
            if (profile.entryDegRatioMin != null) {
              const observed = entryResult.observedEdge ?? opp.edge ?? 0;
              const degRatio =
                (entryResult.capturableEdgeBeforeImpact ?? 0) / Math.max(0.0001, observed);
              if (degRatio < profile.entryDegRatioMin) {
                recordRejection(profile.profileId, "entry_degratio");
                recordEntryDecision(
                  profile.profileId,
                  opp.opportunityId,
                  cycleBucket,
                  false,
                  "entry_degratio",
                  entryResult.capturableEdgeBeforeImpact
                );
                recordEntryChallengerDecision(
                  profile.profileId,
                  opp.opportunityId,
                  cycleBucket,
                  false,
                  "entry_degratio",
                  entryResult.capturableEdgeBeforeImpact,
                  observed,
                  fillRatio ?? undefined,
                  pairKey ?? undefined,
                  degRatio
                );
                annotateBaselineTradeChallengerRejection(
                  BASELINE_PROFILE_ID,
                  opp.opportunityId,
                  "degratioFilteredSameCycle"
                );
                continue;
              }
            }

            recordEntryDecision(
              profile.profileId,
              opp.opportunityId,
              cycleBucket,
              true,
              undefined,
              entryResult.capturableEdgeBeforeImpact
            );
            const degRatioForRecord =
              (entryResult.capturableEdgeBeforeImpact ?? 0) /
              Math.max(0.0001, entryResult.observedEdge ?? opp.edge ?? 0);
            recordEntryChallengerDecision(
              profile.profileId,
              opp.opportunityId,
              cycleBucket,
              true,
              undefined,
              entryResult.capturableEdgeBeforeImpact,
              entryResult.observedEdge,
              fillRatio ?? undefined,
              pairKey ?? undefined,
              degRatioForRecord
            );
            recordFinalCandidate(profile.profileId);
            recordOpenAttempt(profile.profileId);
            const tradeId = `sst-${profile.profileId}-${Date.now()}-${opened}`;
            const narrowTarget = profile.narrowChallengerTarget;
            const structuralTarget = profile.structuralChallengerTarget;
            let effectiveFilledCapital = entryResult.filledCapital;
            let effectiveRequestedCapital = requestedCapital || 0;
            let capitalMultiplier = 1;
            if (structuralRiskTarget) {
              const cap = entryResult.capturableEdgeBeforeImpact ?? 0;
              const observed = entryResult.observedEdge ?? opp.edge ?? 0;
              const degRatio = cap / Math.max(0.0001, observed);
              capitalMultiplier = 1;
              if (cap < 0.055) capitalMultiplier *= 0.6;
              if (degRatio < 0.26) capitalMultiplier *= 0.7;
              capitalMultiplier = Math.max(0.25, Math.min(1, capitalMultiplier));
              effectiveFilledCapital = Math.max(MIN_FILLED_CAPITAL_USD, entryResult.filledCapital * capitalMultiplier);
              effectiveRequestedCapital = (requestedCapital || 0) * capitalMultiplier;
            }
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
              filledCapital: effectiveFilledCapital,
              requestedCapital: effectiveRequestedCapital || requestedCapital || undefined,
              fillRatio: fillRatio ?? null,
              entryImpactBps:
                entryResult.entrySlippage != null
                  ? Math.round(entryResult.entrySlippage * 10000)
                  : null,
              pairKey: makePairKey(opp.marketsInvolved ?? []) || null,
              realizedPnL: 0,
              realizedReturn: 0,
              holdingTimeMs: 0,
              ...(narrowTarget
                ? {
                    narrowTargetPairKeyAtOpen: narrowTarget.pairKey,
                    narrowTargetEdgeBucketAtOpen: narrowTarget.capturableEdgeBucket,
                    narrowTargetFillBucketAtOpen: narrowTarget.fillRatioBucket,
                    narrowFilterMatchAtOpen: true,
                    narrowTargetVersion: "v1",
                  }
                : {}),
              ...(structuralTarget
                ? {
                    structuralTargetPairSetAtOpen: [...structuralTarget.pairKeys],
                    structuralTargetFillBucketAtOpen: structuralTarget.fillRatioBucket,
                    structuralTargetEdgeBucketAtOpen: structuralTarget.capturableEdgeBucket,
                    structuralObservedEdgeBucketAtOpen: getCapturableEdgeBucket(
                      entryResult.capturableEdgeBeforeImpact ?? 0
                    ),
                    structuralFilterMatchAtOpen: true,
                    structuralTargetVersion: "v1",
                  }
                : {}),
              ...(structuralRiskTarget
                ? {
                    structuralRiskCapitalMultiplierAtOpen: capitalMultiplier,
                    structuralRiskCapfloorAtOpen: structuralRiskTarget.capfloor,
                    structuralRiskDegRatioAtOpen:
                      (entryResult.capturableEdgeBeforeImpact ?? 0) /
                      Math.max(0.0001, entryResult.observedEdge ?? opp.edge ?? 0),
                    structuralRiskTargetPairSetAtOpen: [...structuralRiskTarget.pairKeys],
                    structuralRiskTargetFillBucketAtOpen: structuralRiskTarget.fillRatioBucket,
                    structuralRiskFilterMatchAtOpen: true,
                    structuralRiskTargetVersion: "v1",
                  }
                : {}),
            };

            addShadowTrade(profile.profileId, trade, profile);
            recordOpened(profile.profileId);
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
            let earlyThesisFailureReason: string | null = null;
            let exitKillReason: string | null = null;
            let exitKillCapturable: number | null = null;
            let exitKillObserved: number | null = null;
            let exitKillDegRatio: number | null = null;
            let exitKillAbsentCycles: number | null = null;
            let lateExitReason: string | null = null;
            let lateExitCapturable: number | null = null;
            let lateExitObserved: number | null = null;

            if (profile.exitKillTarget && holdingMs < profile.exitKillTarget.monitoringWindowMs) {
              const ek = profile.exitKillTarget;
              if (!latestState) {
                const prev = structuralRiskConsecutiveAbsentByTradeId.get(t.tradeId) ?? 0;
                structuralRiskConsecutiveAbsentByTradeId.set(t.tradeId, prev + 1);
                if (prev + 1 >= ek.killAbsentCycles) {
                  shouldClose = true;
                  exitKillReason = "opportunity_absent";
                  exitKillAbsentCycles = prev + 1;
                  structuralRiskConsecutiveAbsentByTradeId.delete(t.tradeId);
                }
              } else {
                structuralRiskConsecutiveAbsentByTradeId.set(t.tradeId, 0);
                const capturableAtEntry = t.capturableEdgeAtEntry ?? 0;
                const observedAtEntry = t.observedEdgeAtEntry ?? 0;
                const observedNow = latestState.edge;
                const capturableNowProxy =
                  observedAtEntry > 0.0001
                    ? capturableAtEntry * (observedNow / observedAtEntry)
                    : observedNow;
                const degRatioNow =
                  observedNow > 0.0001 ? capturableNowProxy / observedNow : 0;

                if (capturableAtEntry > 0.0001 && capturableNowProxy <= ek.killCapturableDecayFraction * capturableAtEntry) {
                  shouldClose = true;
                  exitKillReason = "capturable_edge_decayed";
                  exitKillCapturable = capturableNowProxy;
                  exitKillObserved = observedNow;
                  exitKillDegRatio = degRatioNow;
                } else if (observedAtEntry > 0.0001 && observedNow <= ek.killObservedEdgeDecayFraction * observedAtEntry) {
                  shouldClose = true;
                  exitKillReason = "observed_edge_decayed";
                  exitKillCapturable = capturableNowProxy;
                  exitKillObserved = observedNow;
                  exitKillDegRatio = degRatioNow;
                } else if (observedNow <= ek.killNetEdgeFloor) {
                  shouldClose = true;
                  exitKillReason = "net_edge_below_floor";
                  exitKillCapturable = capturableNowProxy;
                  exitKillObserved = observedNow;
                  exitKillDegRatio = degRatioNow;
                }
              }
              if (exitKillReason) recordExitKillKilled();
            } else if (
              profile.lateExitTarget &&
              holdingMs >= profile.lateExitTarget.minObservationMs &&
              holdingMs < profile.maxHoldingTimeMs
            ) {
              const le = profile.lateExitTarget;
              if (!latestState) {
                const prev = structuralRiskConsecutiveAbsentByTradeId.get(t.tradeId) ?? 0;
                structuralRiskConsecutiveAbsentByTradeId.set(t.tradeId, prev + 1);
                if (prev + 1 >= le.absentCyclesInLatePhase) {
                  shouldClose = true;
                  lateExitReason = "opportunity_absent_late";
                  structuralRiskConsecutiveAbsentByTradeId.delete(t.tradeId);
                }
              } else {
                structuralRiskConsecutiveAbsentByTradeId.set(t.tradeId, 0);
                const capturableAtEntry = t.capturableEdgeAtEntry ?? 0;
                const observedAtEntry = t.observedEdgeAtEntry ?? 0;
                const observedNow = latestState.edge;
                const capturableNowProxy =
                  observedAtEntry > 0.0001
                    ? capturableAtEntry * (observedNow / observedAtEntry)
                    : observedNow;

                if (observedAtEntry > 0.0001 && observedNow < le.reversionMinFraction * observedAtEntry) {
                  shouldClose = true;
                  lateExitReason = "non_reversion";
                  lateExitCapturable = capturableNowProxy;
                  lateExitObserved = observedNow;
                } else if (observedNow <= le.netEdgeProlongedFloor) {
                  shouldClose = true;
                  lateExitReason = "net_edge_prolonged_low";
                  lateExitCapturable = capturableNowProxy;
                  lateExitObserved = observedNow;
                } else if (observedNow <= le.stagnantEdgeFloor) {
                  const prev = lateExitStagnantCyclesByTradeId.get(t.tradeId) ?? 0;
                  lateExitStagnantCyclesByTradeId.set(t.tradeId, prev + 1);
                  if (prev + 1 >= le.stagnantCycles) {
                    shouldClose = true;
                    lateExitReason = "stagnant_edge";
                    lateExitCapturable = capturableNowProxy;
                    lateExitObserved = observedNow;
                    lateExitStagnantCyclesByTradeId.delete(t.tradeId);
                  }
                } else {
                  lateExitStagnantCyclesByTradeId.set(t.tradeId, 0);
                }
              }
            } else if (
              profile.structuralRiskManagedTarget &&
              !profile.exitKillTarget &&
              !profile.lateExitTarget &&
              holdingMs < EARLY_THESIS_MONITORING_WINDOW_MS
            ) {
              if (!latestState) {
                const prev = structuralRiskConsecutiveAbsentByTradeId.get(t.tradeId) ?? 0;
                structuralRiskConsecutiveAbsentByTradeId.set(t.tradeId, prev + 1);
                if (prev + 1 >= 2) {
                  shouldClose = true;
                  earlyThesisFailureReason = "structural_opportunity_disappeared_2_cycles";
                  structuralRiskConsecutiveAbsentByTradeId.delete(t.tradeId);
                }
              } else {
                structuralRiskConsecutiveAbsentByTradeId.set(t.tradeId, 0);
                const capturableAtEntry = t.capturableEdgeAtEntry ?? 0;
                const observedAtEntry = t.observedEdgeAtEntry ?? 0;
                const observedNow = latestState.edge;
                const capturableNowProxy =
                  observedAtEntry > 0.0001
                    ? capturableAtEntry * (observedNow / observedAtEntry)
                    : observedNow;
                if (capturableAtEntry > 0.0001 && capturableNowProxy <= 0.6 * capturableAtEntry) {
                  shouldClose = true;
                  earlyThesisFailureReason = "capturable_edge_decayed";
                } else if (observedNow <= 0.5 * (profile.structuralRiskManagedTarget?.capfloor ?? 0.045)) {
                  shouldClose = true;
                  earlyThesisFailureReason = "net_edge_below_half_threshold";
                }
              }
            }

            if (!shouldClose) {
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

              const closeUpdates: Parameters<typeof closeShadowTrade>[2] = {
                closedAt: exitResult.exitLatencyMs > 0 ? new Date(now + exitResult.exitLatencyMs).toISOString() : new Date().toISOString(),
                effectiveExitPrice: exitResult.effectiveExitPrice,
                realizedPnL: exitResult.realizedPnL,
                realizedReturn: exitResult.realizedReturn,
                holdingTimeMs: holdingMs,
                exitReason: lateExitReason
                  ? "late_exit"
                  : exitKillReason
                    ? "exit_kill"
                    : earlyThesisFailureReason
                      ? "early_thesis_failure"
                      : exitResult.exitReason,
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
                  exitReason: earlyThesisFailureReason ?? exitResult.exitReason,
                  edgeAtExit: edgeAtExit ?? undefined,
                },
              };
              if (earlyThesisFailureReason) {
                closeUpdates.earlyThesisFailureTriggered = true;
                closeUpdates.earlyThesisFailureReason = earlyThesisFailureReason;
                closeUpdates.earlyThesisFailureAtMsFromOpen = holdingMs;
              }
              if (exitKillReason) {
                closeUpdates.exitKillTriggered = true;
                closeUpdates.exitKillReason = exitKillReason;
                closeUpdates.exitKillAtMsFromOpen = holdingMs;
                closeUpdates.capturableEdgeAtKill = exitKillCapturable;
                closeUpdates.observedEdgeAtKill = exitKillObserved;
                closeUpdates.degradationRatioAtKill = exitKillDegRatio;
                closeUpdates.opportunityAbsentCyclesAtKill = exitKillAbsentCycles;
              }
              if (lateExitReason) {
                closeUpdates.lateExitTriggered = true;
                closeUpdates.lateExitReason = lateExitReason;
                closeUpdates.lateExitAtMsFromOpen = holdingMs;
                closeUpdates.capturableEdgeAtLateExit = lateExitCapturable;
                closeUpdates.observedEdgeAtLateExit = lateExitObserved;
              }
              closeShadowTrade(profile.profileId, t.tradeId, closeUpdates);
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
          updateProfileHeartbeat(profile.profileId);
        } catch (err) {
          console.warn(`[ShadowSim] profile ${profile.profileId} failed:`, err instanceof Error ? err.message : err);
          updateProfileHeartbeat(profile.profileId);
        }
      }

      lastUpdateMs = Date.now();
      lastCycleOk = true;
      recordShadowLoopCompleted();
    })
    .catch((err) => {
      lastCycleOk = false;
      recordShadowLoopCompleted();
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
        if (!isEdgeBucketMatch(entryResult.capturableEdgeBeforeImpact ?? 0, target.capturableEdgeBucket)) {
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
      // Structural challenger: pair set + fill bucket + edge bucket opcional
      if (profile.structuralChallengerTarget) {
        recordStructuralOpportunitiesSeen(profile.profileId);
        if (!isStructuralPairMatch(pairKey ?? null)) {
          recordStructuralRejectedByPairMismatch(profile.profileId);
          recordRejection(profile.profileId, "structural_pair_mismatch");
          continue;
        }
        if (!isStructuralFillBucketMatch(fillRatio)) {
          recordStructuralRejectedByFillBucketMismatch(profile.profileId);
          recordRejection(profile.profileId, "structural_fill_bucket_mismatch");
          continue;
        }
        if (profile.structuralChallengerTarget.capturableEdgeBucket === ">5%") {
          if (!isStructuralEdgeBucketGt5Match(entryResult.capturableEdgeBeforeImpact ?? 0)) {
            recordStructuralRejectedByEdgeBucketMismatch(profile.profileId);
            recordRejection(profile.profileId, "structural_edge_bucket_mismatch");
            continue;
          }
        }
        recordStructuralPassedAllFilters(profile.profileId);
      }
      // Structural risk-managed: pair + fill + capfloor 4.5% + degratio 0.24
      const structuralRiskTargetEval = profile.structuralRiskManagedTarget;
      if (structuralRiskTargetEval) {
        recordStructuralRiskEvaluated();
        if (!isStructuralPairMatch(pairKey ?? null)) {
          recordStructuralRiskRejectedByPair();
          recordRejection(profile.profileId, "structural_risk_pair_mismatch");
          continue;
        }
        if (!isStructuralFillBucketMatch(fillRatio)) {
          recordStructuralRiskRejectedByFillBucket();
          recordRejection(profile.profileId, "structural_risk_fill_bucket_mismatch");
          continue;
        }
        const capEval = entryResult.capturableEdgeBeforeImpact ?? 0;
        if (capEval < structuralRiskTargetEval.capfloor) {
          recordStructuralRiskRejectedByCapfloor();
          recordRejection(profile.profileId, "structural_risk_capfloor");
          continue;
        }
        const observedEval = entryResult.observedEdge ?? opp.edge ?? 0;
        const degRatioEval = capEval / Math.max(0.0001, observedEval);
        if (degRatioEval < structuralRiskTargetEval.degRatioMin) {
          recordStructuralRiskRejectedByDegRatio();
          recordRejection(profile.profileId, "structural_risk_degratio");
          continue;
        }
      }
      // Entry capfloor challenger: só abre se capturableEdge >= 0.03
      if (profile.entryCapfloorMinCapturableEdge != null) {
        const cap = entryResult.capturableEdgeBeforeImpact ?? 0;
        if (cap < profile.entryCapfloorMinCapturableEdge) {
          recordRejection(profile.profileId, "entry_capfloor");
          continue;
        }
      }
      // Entry degratio challenger: só abre se capturable/observed >= 0.22
      if (profile.entryDegRatioMin != null) {
        const observed = entryResult.observedEdge ?? opp.edge ?? 0;
        const degRatio = (entryResult.capturableEdgeBeforeImpact ?? 0) / Math.max(0.0001, observed);
        if (degRatio < profile.entryDegRatioMin) {
          recordRejection(profile.profileId, "entry_degratio");
          continue;
        }
      }
      const narrowTarget = profile.narrowChallengerTarget;
      const structuralTarget = profile.structuralChallengerTarget;
      const structuralRiskTargetEval2 = profile.structuralRiskManagedTarget;
      let effectiveFilledCapitalEval = entryResult.filledCapital;
      let effectiveRequestedCapitalEval = requestedCapital || 0;
      let capitalMultiplierEval = 1;
      if (structuralRiskTargetEval2) {
        const capE = entryResult.capturableEdgeBeforeImpact ?? 0;
        const obsE = entryResult.observedEdge ?? opp.edge ?? 0;
        const degE = capE / Math.max(0.0001, obsE);
        capitalMultiplierEval = 1;
        if (capE < 0.055) capitalMultiplierEval *= 0.6;
        if (degE < 0.26) capitalMultiplierEval *= 0.7;
        capitalMultiplierEval = Math.max(0.25, Math.min(1, capitalMultiplierEval));
        effectiveFilledCapitalEval = Math.max(MIN_FILLED_CAPITAL_USD, entryResult.filledCapital * capitalMultiplierEval);
        effectiveRequestedCapitalEval = (requestedCapital || 0) * capitalMultiplierEval;
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
        filledCapital: effectiveFilledCapitalEval,
        requestedCapital: effectiveRequestedCapitalEval || requestedCapital || undefined,
        fillRatio: fillRatio ?? null,
        entryImpactBps:
          entryResult.entrySlippage != null
            ? Math.round(entryResult.entrySlippage * 10000)
            : null,
        pairKey: makePairKey(opp.marketsInvolved ?? []) || null,
        realizedPnL: 0,
        realizedReturn: 0,
        holdingTimeMs: 0,
        ...(narrowTarget
          ? {
              narrowTargetPairKeyAtOpen: narrowTarget.pairKey,
              narrowTargetEdgeBucketAtOpen: narrowTarget.capturableEdgeBucket,
              narrowTargetFillBucketAtOpen: narrowTarget.fillRatioBucket,
              narrowFilterMatchAtOpen: true,
              narrowTargetVersion: "v1",
            }
          : {}),
        ...(structuralTarget
          ? {
              structuralTargetPairSetAtOpen: [...structuralTarget.pairKeys],
              structuralTargetFillBucketAtOpen: structuralTarget.fillRatioBucket,
              structuralTargetEdgeBucketAtOpen: structuralTarget.capturableEdgeBucket,
              structuralObservedEdgeBucketAtOpen: getCapturableEdgeBucket(
                entryResult.capturableEdgeBeforeImpact ?? 0
              ),
              structuralFilterMatchAtOpen: true,
              structuralTargetVersion: "v1",
            }
          : {}),
        ...(structuralRiskTargetEval2
          ? {
              structuralRiskCapitalMultiplierAtOpen: capitalMultiplierEval,
              structuralRiskCapfloorAtOpen: structuralRiskTargetEval2.capfloor,
              structuralRiskDegRatioAtOpen:
                (entryResult.capturableEdgeBeforeImpact ?? 0) /
                Math.max(0.0001, entryResult.observedEdge ?? opp.edge ?? 0),
              structuralRiskTargetPairSetAtOpen: [...structuralRiskTargetEval2.pairKeys],
              structuralRiskTargetFillBucketAtOpen: structuralRiskTargetEval2.fillRatioBucket,
              structuralRiskFilterMatchAtOpen: true,
              structuralRiskTargetVersion: "v1",
            }
          : {}),
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
