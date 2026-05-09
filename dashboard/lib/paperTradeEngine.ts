/**
 * Paper Trade Engine — manages lifecycle of simulated trades.
 * Opens, updates, closes paper trades based on opportunities and exit conditions.
 */

import type { NormalizedPaperOpportunity } from "./paperTypes";
import type { CapacityResult } from "./capitalCapacityEngine";
import {
  simulateEntry,
  simulateExit,
  shouldClosePaperTrade,
  resolvePaperExitSafety,
  type SimulatedEntry,
  type SimulatedExit,
  type ActivePaperTradeState,
} from "./executionSimulator";
import { getPaperDynamicExitConfig } from "./paperDynamicExitConfig";
import {
  applyDynamicMetricsToTrade,
  evaluateDynamicExit,
  buildExitDecisionSnapshot,
} from "./paperDynamicExit";
import type { PaperExitDecisionSnapshot, PaperTrade } from "./paperTypes";
import { getAllMarkets } from "./marketDataService";
import { resolveMarkPxFromTrade } from "./paperMarkToMarket";
import {
  getPaperPortfolio,
  addActiveTrade,
  closeTrade,
  getActivePaperTrades,
  updateActiveTradeMtm,
  getActiveTradeById,
} from "./paperPortfolioStore";
import { logTradeRejection } from "./tradeRejectionLogger";
import {
  recordPaperEngineBatch,
  bumpPaperOpenRejection,
  bumpPaperReachedPreSimulate,
  bumpPaperOpened,
  recordPaperNetEdgeThresholdRejection,
  recordPaperEconomicEntryOutcome,
  recordPaperEconomicCandidateEvaluation,
  recordPaperCooldownSuppress,
  recordPaperOpenedThenNoProgressExit,
  recordPaperEntryEconomicScoreAtClose,
  recordPaperUpstreamFamilySeen,
  setUpstreamSelectionLastCycle,
  getFamilyRecentCount,
} from "./paperOpenDiagnostics";
import {
  recordGraphEngineBatchIfApplicable,
  recordGraphPassedEconomicIfApplicable,
  recordGraphRejectEconomic,
  recordGraphRejectPreEconomic,
  recordGraphTradeClosed,
  recordGraphTradeOpened,
  inferPaperGraphProvenanceFromOpportunityType,
  normalizePaperGraphProvenance,
} from "./graphOpportunityPaperImpact";
import {
  applyUpstreamDiversitySelection,
  getOpportunityFamilyKey,
  getUpstreamDiversityPolicySnapshot,
} from "./paperUpstreamDiversity";
import {
  isEconomicCooldownActive,
  recordEconomicOutcomeForCooldown,
  makeEconomicDedupeKey,
} from "./paperEconomicCooldown";
import {
  getPaperEntryEconomicFilterConfig,
  evaluatePaperEntryEconomics,
  computePaperEntryEconomicsMetrics,
  collectAllPaperEntryEconomicsFailures,
  computeProgressProbabilityInternals,
  STANDARD_CROSS_MARKET_PROFILE_KEY,
  type PaperEntryEconomicFilterConfig,
  type PaperEntryEconomicsMetrics,
} from "./paperEntryEconomics";
import {
  entryProfileKey,
  getEffectiveMinProgressProbabilityFactor,
  getHistoricalNoProgressRate,
  getHistoricalTakeProfitRate,
  recordPaperExitProfileOutcome,
} from "./paperEntryProfileMemory";
import { isOpportunityClassBlocked } from "./paperProtectedCore";
import {
  recordPaperTradeLifecycleTick,
  recordPaperTradeLifecycleClose,
} from "./paperTradeLifecycleDiagnostics";

export interface PaperTradePolicy {
  initialCapital: number;
  minConfidenceToTrade: number;
  minNetEdgeToTrade: number;
  maxCapitalPerTrade: number;
  maxCapitalPerCluster: number;
  maxCapitalPerMarket: number;
  maxHoldingTimeMs: number;
  stopLossPct: number;
  takeProfitPct: number;
  feeBuffer: number;
  edgeDecayFactor: number;
  edgeNormalizationThreshold: number;
  /** Fecho por captura: edge corrente caiu ≥ este delta vs grossEdgeAtEntry (absoluto). */
  edgeCaptureDelta: number;
  /** Fecho por deterioração: edge corrente ≤ entrada − delta, ou edge < 0. */
  edgeDeteriorationDelta: number;
  minEpisodeDurationMs?: number;
  /** USD por trade quando > 0 (PAPER_FIXED_TRADE_USD); senão usa capacity.recommendedCapital. */
  fixedTradeSizeUsd?: number;
}

export const DEFAULT_PAPER_POLICY: PaperTradePolicy = {
  initialCapital: 10_000,
  minConfidenceToTrade: 0.15,
  minNetEdgeToTrade: 0.008,
  maxCapitalPerTrade: 500,
  maxCapitalPerCluster: 1000,
  maxCapitalPerMarket: 300,
  maxHoldingTimeMs: 180_000,
  stopLossPct: 0.03,
  takeProfitPct: 0.01,
  feeBuffer: 0.002,
  edgeDecayFactor: 1.5,
  edgeNormalizationThreshold: 0.005,
  edgeCaptureDelta: 0.004,
  edgeDeteriorationDelta: 0.015,
  minEpisodeDurationMs: 5000,
};

/** Policy paper com PAPER_FIXED_TRADE_USD opcional (número > 0). */
export function resolvePaperPolicyFromEnv(): PaperTradePolicy {
  const pol: PaperTradePolicy = { ...DEFAULT_PAPER_POLICY };
  const raw = process.env.PAPER_FIXED_TRADE_USD?.trim();
  if (raw) {
    const n = Number(raw);
    if (Number.isFinite(n) && n > 0) pol.fixedTradeSizeUsd = n;
  }
  const rawMinNet = process.env.PAPER_MIN_NET_EDGE_TO_TRADE?.trim();
  if (rawMinNet) {
    const v = Number(rawMinNet);
    if (Number.isFinite(v) && v >= 0 && v <= 1) pol.minNetEdgeToTrade = v;
  }
  return pol;
}

/** Snapshot só para APIs / diagnóstico (sem efeito na execução). */
export function getPaperEntryPolicySnapshot(): {
  minNetEdgeToTradeEffective: number;
  minNetEdgeToTradeDefault: number;
  paperMinNetEdgeToTradeEnvRaw: string | null;
  economicEntryFilter: PaperEntryEconomicFilterConfig;
} {
  const pol = resolvePaperPolicyFromEnv();
  const raw = process.env.PAPER_MIN_NET_EDGE_TO_TRADE?.trim();
  return {
    minNetEdgeToTradeEffective: pol.minNetEdgeToTrade,
    minNetEdgeToTradeDefault: DEFAULT_PAPER_POLICY.minNetEdgeToTrade,
    paperMinNetEdgeToTradeEnvRaw: raw && raw.length > 0 ? raw : null,
    economicEntryFilter: getPaperEntryEconomicFilterConfig(),
  };
}

let tradeSeq = 0;

function makeTradeId(): string {
  return `pt-${++tradeSeq}-${Date.now()}`;
}

function hasActiveTradeForOpportunity(opportunityId: string): boolean {
  return getActivePaperTrades().some((t) => t.opportunityId === opportunityId);
}

function toActiveState(t: {
  tradeId: string;
  opportunityId: string;
  openedAt: string;
  entryPriceEstimate: number;
  filledCapital: number;
  grossEdgeAtEntry: number;
  maxAdverseExcursion?: number;
  maxFavorableExcursion?: number;
}): ActivePaperTradeState {
  return {
    tradeId: t.tradeId,
    opportunityId: t.opportunityId,
    openedAt: t.openedAt,
    entryEdge: t.grossEdgeAtEntry,
    entryGrossEdge: t.grossEdgeAtEntry,
    entryPriceEstimate: t.entryPriceEstimate,
    filledCapital: t.filledCapital,
    maxAdverseExcursion: t.maxAdverseExcursion ?? 0,
    maxFavorableExcursion: t.maxFavorableExcursion ?? 0,
  };
}

export function processOpportunities(
  opportunities: Array<{ opp: NormalizedPaperOpportunity; capacity: CapacityResult }>,
  policy: Partial<PaperTradePolicy> = {}
): { opened: number; closed: number; rejected: number } {
  const pol = { ...DEFAULT_PAPER_POLICY, ...policy };
  const portfolio = getPaperPortfolio();
  const divPol = getUpstreamDiversityPolicySnapshot();
  let batch = opportunities;
  if (divPol.enabled) {
    const { selected, diagnostics } = applyUpstreamDiversitySelection(
      opportunities,
      getFamilyRecentCount
    );
    setUpstreamSelectionLastCycle(diagnostics);
    batch = selected;
  } else {
    setUpstreamSelectionLastCycle(null);
  }

  const oppMap = new Map(batch.map((o) => [o.opp.opportunityId, o]));
  const dynCfg = getPaperDynamicExitConfig(pol.maxHoldingTimeMs);

  let opened = 0;
  let closed = 0;
  let rejected = 0;

  recordPaperEngineBatch(batch.length);

  for (const { opp, capacity } of batch) {
    recordPaperUpstreamFamilySeen(getOpportunityFamilyKey(opp));
    recordGraphEngineBatchIfApplicable(opp);
    const marketId = opp.marketsInvolved[0]?.marketId ?? opp.opportunityId;
    const edgeBps = Math.round((capacity.estimatedNetEdge ?? opp.edge) * 10000);

    if (capacity.recommendedCapital <= 0) {
      bumpPaperOpenRejection("RECOMMENDED_CAPITAL_LE_ZERO_AT_ENGINE");
      recordGraphRejectPreEconomic(opp, "rec_cap_le0");
      logTradeRejection({ timestamp: Date.now(), marketId, edgeBps, reason: "EDGE_BELOW_THRESHOLD" });
      rejected++;
      continue;
    }
    if (capacity.estimatedNetEdge < pol.minNetEdgeToTrade) {
      recordPaperNetEdgeThresholdRejection(capacity.estimatedNetEdge);
      bumpPaperOpenRejection("ESTIMATED_NET_EDGE_BELOW_THRESHOLD");
      recordGraphRejectPreEconomic(opp, "net_edge_below_threshold");
      logTradeRejection({ timestamp: Date.now(), marketId, edgeBps, reason: "EDGE_BELOW_THRESHOLD" });
      rejected++;
      continue;
    }
    if (opp.confidence < pol.minConfidenceToTrade) {
      bumpPaperOpenRejection("CONFIDENCE_BELOW_THRESHOLD");
      recordGraphRejectPreEconomic(opp, "confidence_below_threshold");
      logTradeRejection({ timestamp: Date.now(), marketId, edgeBps, reason: "LATENCY_RISK" });
      rejected++;
      continue;
    }
    if (hasActiveTradeForOpportunity(opp.opportunityId)) {
      bumpPaperOpenRejection("ACTIVE_TRADE_FOR_OPPORTUNITY");
      recordGraphRejectPreEconomic(opp, "active_duplicate");
      continue;
    }

    const safetyBlockReason = isOpportunityClassBlocked(opp);
    if (safetyBlockReason) {
      bumpPaperOpenRejection("BLOCKED_BY_SAFETY_GATE");
      recordGraphRejectPreEconomic(opp, "safety_gate_blocked");
      logTradeRejection({ timestamp: Date.now(), marketId, edgeBps, reason: "EDGE_BELOW_THRESHOLD" });
      rejected++;
      continue;
    }

    const clusterExposure = (opp.clusterId && portfolio.exposureByCluster[opp.clusterId]) || 0;
    const marketExposure = opp.marketsInvolved.reduce(
      (s, m) => s + (portfolio.exposureByMarket[m.marketId] || 0),
      0
    );
    if (clusterExposure >= pol.maxCapitalPerCluster || marketExposure >= pol.maxCapitalPerMarket) {
      bumpPaperOpenRejection("EXPOSURE_CLUSTER_OR_MARKET_LIMIT");
      recordGraphRejectPreEconomic(opp, "exposure_limit");
      logTradeRejection({ timestamp: Date.now(), marketId, edgeBps, reason: "TRADE_SIZE_TOO_SMALL" });
      rejected++;
      continue;
    }

    const fixed = pol.fixedTradeSizeUsd != null && pol.fixedTradeSizeUsd > 0;
    const requested = fixed
      ? Math.min(
          pol.fixedTradeSizeUsd!,
          pol.maxCapitalPerTrade,
          portfolio.availableCapital,
          Math.max(0, opp.liquidity * 0.1)
        )
      : Math.min(capacity.recommendedCapital, pol.maxCapitalPerTrade, portfolio.availableCapital);
    if (requested <= 0) {
      bumpPaperOpenRejection("REQUESTED_CAPITAL_LE_ZERO");
      recordGraphRejectPreEconomic(opp, "requested_cap_le0");
      logTradeRejection({ timestamp: Date.now(), marketId, edgeBps, reason: "TRADE_SIZE_TOO_SMALL" });
      rejected++;
      continue;
    }

    const profileKey = entryProfileKey(opp.sourceType, opp.opportunityType);
    const dedupeKeyForCooldown = makeEconomicDedupeKey(profileKey, opp.opportunityId);
    if (isEconomicCooldownActive(dedupeKeyForCooldown)) {
      bumpPaperOpenRejection("SUPPRESSED_BY_ECONOMIC_COOLDOWN");
      recordGraphRejectPreEconomic(opp, "economic_cooldown");
      recordPaperCooldownSuppress(profileKey, dedupeKeyForCooldown);
      rejected++;
      continue;
    }

    bumpPaperReachedPreSimulate();
    const entry: SimulatedEntry = simulateEntry(opp, capacity, portfolio.availableCapital, {
      requestedCapital: requested,
    });
    if (entry.filledCapital <= 0) {
      bumpPaperOpenRejection("SIMULATOR_FILLED_CAPITAL_LE_ZERO");
      recordGraphRejectPreEconomic(opp, "simulator_fill_le0");
      logTradeRejection({ timestamp: Date.now(), marketId, edgeBps, reason: "INSUFFICIENT_LIQUIDITY" });
      rejected++;
      continue;
    }

    const econCfg = getPaperEntryEconomicFilterConfig();
    let entryMetricsForOpen: PaperEntryEconomicsMetrics | undefined;
    if (econCfg.enabled) {
      const ev = evaluatePaperEntryEconomics(
        capacity,
        entry,
        pol.feeBuffer,
        pol.minNetEdgeToTrade,
        profileKey,
        econCfg
      );
      entryMetricsForOpen = ev.metrics;
      recordPaperEconomicEntryOutcome(ev.metrics, ev.ok);
      const reasonsAll = collectAllPaperEntryEconomicsFailures(ev.metrics, profileKey, econCfg);
      const internals = computeProgressProbabilityInternals(
        capacity,
        entry,
        ev.metrics.grossToFeesRatio,
        pol.minNetEdgeToTrade,
        ev.metrics.historicalNoProgressPrior,
        econCfg
      );
      const progressGuardRow = getEffectiveMinProgressProbabilityFactor(
        profileKey,
        econCfg.minProgressProbabilityFactorToOpen,
        {
          enableAdaptive: econCfg.enableAdaptiveProgressGuard,
          minSamples: econCfg.minSamplesForAdaptiveProgressGuard,
          extraMax: econCfg.adaptiveProgressGuardExtraMax,
        }
      );
      const crossApplicable =
        profileKey === STANDARD_CROSS_MARKET_PROFILE_KEY && econCfg.enableCrossMarketNetGrossEntryGuard;
      const minNetGrossCm = crossApplicable ? econCfg.minNetToGrossEdgeRatioCrossMarket : null;
      recordPaperEconomicCandidateEvaluation({
        evaluatedAt: new Date().toISOString(),
        opportunityId: opp.opportunityId,
        marketLabel: opp.marketsInvolved[0]?.question
          ? String(opp.marketsInvolved[0].question).slice(0, 160)
          : null,
        sourceType: opp.sourceType,
        opportunityType: opp.opportunityType,
        profileKey,
        confidence: opp.confidence,
        fillProbability: entry.fillProbability,
        capacityConfidence: capacity.capacityConfidence,
        recommendedCapital: capacity.recommendedCapital,
        requestedCapital: requested,
        filledCapital: entry.filledCapital,
        estimatedGrossEdge: capacity.estimatedGrossEdge,
        estimatedNetEdge: capacity.estimatedNetEdge,
        grossEdgeAtEntry: opp.edge,
        netEdgeAtEntry: capacity.estimatedNetEdge,
        expectedGrossPnlUsd: ev.metrics.expectedGrossPnlToOpenUsd,
        expectedFeesUsd: ev.metrics.expectedFeesUsd,
        expectedNetPnlToOpenUsd: ev.metrics.expectedNetPnlToOpenUsd,
        expectedRealizableNetPnlUsd: ev.metrics.expectedRealizableNetPnlUsd,
        expectedNetProfitMargin: ev.metrics.expectedNetProfitMargin,
        grossToFeesRatio: ev.metrics.grossToFeesRatio,
        liquiditySignal: internals.liquiditySignal,
        headroomFactor: internals.headroomFactor,
        monetizationFactor: internals.monetizationFactor,
        historicalNoProgressFactor: internals.historicalNoProgressFactor,
        progressProbabilityFactor: ev.metrics.progressProbabilityFactor,
        progressProbabilityFactorEffectiveThreshold:
          ev.metrics.effectiveMinProgressProbabilityFactorToOpen ?? econCfg.minProgressProbabilityFactorToOpen,
        entryEconomicScore: ev.metrics.entryEconomicScore,
        minEntryEconomicScoreEffective: econCfg.minEntryEconomicScore,
        netToGrossEdgeRatioAtEntry: ev.metrics.netToGrossEdgeRatioAtEntry,
        minNetToGrossEdgeRatioCrossMarket: minNetGrossCm,
        passedEconomicFilters: ev.ok,
        finalEconomicDecision: ev.ok ? "pass" : "fail",
        rejectionReasonFinal: ev.ok ? null : ev.reason,
        rejectionReasonsAll: reasonsAll,
        crossMarketGuardApplicable: crossApplicable,
        adaptiveProgressGuardApplied: ev.metrics.adaptiveProgressGuardApplied ?? false,
        adaptiveProfileStress: progressGuardRow.adaptiveApplied ? progressGuardRow.stress : null,
        historicalNoProgressRate: getHistoricalNoProgressRate(profileKey, econCfg.minSamplesForHistoricalPrior),
        historicalTakeProfitRate: getHistoricalTakeProfitRate(profileKey, econCfg.minSamplesForHistoricalPrior),
        economicEntryFilterEnabled: true,
      });
      recordEconomicOutcomeForCooldown(
        dedupeKeyForCooldown,
        ev.ok,
        ev.ok ? null : ev.reason,
        reasonsAll
      );
      if (!ev.ok) {
        bumpPaperOpenRejection(ev.reason);
        recordGraphRejectEconomic(opp, String(ev.reason));
        logTradeRejection({ timestamp: Date.now(), marketId, edgeBps, reason: "EDGE_BELOW_THRESHOLD" });
        rejected++;
        continue;
      }
      recordGraphPassedEconomicIfApplicable(opp);
    } else {
      const base = computePaperEntryEconomicsMetrics(
        capacity,
        entry,
        pol.feeBuffer,
        pol.minNetEdgeToTrade,
        profileKey
      );
      const gr = getEffectiveMinProgressProbabilityFactor(profileKey, econCfg.minProgressProbabilityFactorToOpen, {
        enableAdaptive: econCfg.enableAdaptiveProgressGuard,
        minSamples: econCfg.minSamplesForAdaptiveProgressGuard,
        extraMax: econCfg.adaptiveProgressGuardExtraMax,
      });
      entryMetricsForOpen = {
        ...base,
        globalMinProgressProbabilityFactorToOpen: econCfg.minProgressProbabilityFactorToOpen,
        effectiveMinProgressProbabilityFactorToOpen: gr.effectiveMin,
        adaptiveProgressGuardApplied: gr.adaptiveApplied,
      };
      recordPaperEconomicEntryOutcome(entryMetricsForOpen, true);
      const internals = computeProgressProbabilityInternals(
        capacity,
        entry,
        entryMetricsForOpen.grossToFeesRatio,
        pol.minNetEdgeToTrade,
        entryMetricsForOpen.historicalNoProgressPrior,
        econCfg
      );
      const crossApplicable =
        profileKey === STANDARD_CROSS_MARKET_PROFILE_KEY && econCfg.enableCrossMarketNetGrossEntryGuard;
      recordPaperEconomicCandidateEvaluation({
        evaluatedAt: new Date().toISOString(),
        opportunityId: opp.opportunityId,
        marketLabel: opp.marketsInvolved[0]?.question
          ? String(opp.marketsInvolved[0].question).slice(0, 160)
          : null,
        sourceType: opp.sourceType,
        opportunityType: opp.opportunityType,
        profileKey,
        confidence: opp.confidence,
        fillProbability: entry.fillProbability,
        capacityConfidence: capacity.capacityConfidence,
        recommendedCapital: capacity.recommendedCapital,
        requestedCapital: requested,
        filledCapital: entry.filledCapital,
        estimatedGrossEdge: capacity.estimatedGrossEdge,
        estimatedNetEdge: capacity.estimatedNetEdge,
        grossEdgeAtEntry: opp.edge,
        netEdgeAtEntry: capacity.estimatedNetEdge,
        expectedGrossPnlUsd: entryMetricsForOpen.expectedGrossPnlToOpenUsd,
        expectedFeesUsd: entryMetricsForOpen.expectedFeesUsd,
        expectedNetPnlToOpenUsd: entryMetricsForOpen.expectedNetPnlToOpenUsd,
        expectedRealizableNetPnlUsd: entryMetricsForOpen.expectedRealizableNetPnlUsd,
        expectedNetProfitMargin: entryMetricsForOpen.expectedNetProfitMargin,
        grossToFeesRatio: entryMetricsForOpen.grossToFeesRatio,
        liquiditySignal: internals.liquiditySignal,
        headroomFactor: internals.headroomFactor,
        monetizationFactor: internals.monetizationFactor,
        historicalNoProgressFactor: internals.historicalNoProgressFactor,
        progressProbabilityFactor: entryMetricsForOpen.progressProbabilityFactor,
        progressProbabilityFactorEffectiveThreshold:
          entryMetricsForOpen.effectiveMinProgressProbabilityFactorToOpen ??
          econCfg.minProgressProbabilityFactorToOpen,
        entryEconomicScore: entryMetricsForOpen.entryEconomicScore,
        minEntryEconomicScoreEffective: econCfg.minEntryEconomicScore,
        netToGrossEdgeRatioAtEntry: entryMetricsForOpen.netToGrossEdgeRatioAtEntry,
        minNetToGrossEdgeRatioCrossMarket: crossApplicable ? econCfg.minNetToGrossEdgeRatioCrossMarket : null,
        passedEconomicFilters: true,
        finalEconomicDecision: "filter_disabled_pass",
        rejectionReasonFinal: null,
        rejectionReasonsAll: [],
        crossMarketGuardApplicable: crossApplicable,
        adaptiveProgressGuardApplied: entryMetricsForOpen.adaptiveProgressGuardApplied ?? false,
        adaptiveProfileStress: gr.adaptiveApplied ? gr.stress : null,
        historicalNoProgressRate: getHistoricalNoProgressRate(profileKey, econCfg.minSamplesForHistoricalPrior),
        historicalTakeProfitRate: getHistoricalTakeProfitRate(profileKey, econCfg.minSamplesForHistoricalPrior),
        economicEntryFilterEnabled: false,
      });
      recordGraphPassedEconomicIfApplicable(opp);
    }

    const tradeId = makeTradeId();
    const netEdge = capacity.estimatedNetEdge;
    addActiveTrade({
      tradeId,
      opportunityId: opp.opportunityId,
      sourceType: opp.sourceType,
      opportunityType: opp.opportunityType,
      clusterId: opp.clusterId,
      marketsInvolved: opp.marketsInvolved,
      openedAt: entry.entryTimestamp,
      closedAt: null,
      status: "active",
      grossEdgeAtEntry: opp.edge,
      netEdgeAtEntry: netEdge,
      recommendedCapital: capacity.recommendedCapital,
      requestedCapital: requested,
      filledCapital: entry.filledCapital,
      entryPriceEstimate: entry.entryPriceEstimate,
      entryConfidence: opp.confidence,
      realizedPnL: 0,
      realizedReturn: 0,
      holdingTimeMs: 0,
      maxAdverseExcursion: 0,
      maxFavorableExcursion: 0,
      notes: entry.partialFillFlag ? "partial_fill" : undefined,
      entryEconomicScoreAtOpen: entryMetricsForOpen?.entryEconomicScore,
      progressProbabilityFactorAtOpen: entryMetricsForOpen?.progressProbabilityFactor,
      entryProfileKeyAtOpen: profileKey,
      graphDiagnosticProvenanceAtOpen:
        opp.sourceType === "graph"
          ? opp.graphDiagnosticProvenance != null &&
            typeof opp.graphDiagnosticProvenance === "string" &&
            opp.graphDiagnosticProvenance.length > 0
            ? normalizePaperGraphProvenance(opp.graphDiagnosticProvenance)
            : inferPaperGraphProvenanceFromOpportunityType(opp.opportunityType)
          : undefined,
      structuralMicroLaneReasonAtOpen:
        opp.sourceType === "graph" ? opp.structuralMicroLaneReason : undefined,
    });
    recordGraphTradeOpened(
      opp,
      entryMetricsForOpen?.entryEconomicScore,
      entryMetricsForOpen?.progressProbabilityFactor
    );
    bumpPaperOpened();
    opened++;
  }

  const marketsById = new Map(getAllMarkets().map((m) => [m.id, m]));
  const active = getActivePaperTrades();
  for (const t of active) {
    const latest = oppMap.get(t.opportunityId);
    let latestState = latest
      ? { edge: latest.opp.edge, confidence: latest.opp.confidence }
      : null;
    let markSource: "opp_map" | "mtm" | "none" = latest ? "opp_map" : "none";

    const markPxMtm = resolveMarkPxFromTrade(t, marketsById);
    if (markPxMtm != null) {
      const entry = t.entryPriceEstimate;
      const ret = (markPxMtm - entry) / Math.max(1e-9, entry);
      const newFav = Math.max(t.maxFavorableExcursion ?? 0, ret);
      const newAdv = Math.max(t.maxAdverseExcursion ?? 0, -ret);
      updateActiveTradeMtm(t.tradeId, {
        lastMarkPx: markPxMtm,
        lastMarkAt: new Date().toISOString(),
        maxAdverseExcursion: newAdv,
        maxFavorableExcursion: newFav,
      });
    }

    if (!latestState && markPxMtm != null) {
      latestState = { edge: 1 - markPxMtm, confidence: t.entryConfidence };
      markSource = "mtm";
    }

    const live = getActiveTradeById(t.tradeId) ?? t;
    const markPxFromLatest = latestState != null ? 1 - latestState.edge : null;
    recordPaperTradeLifecycleTick({
      tradeId: t.tradeId,
      opportunityId: t.opportunityId,
      opportunityType: t.opportunityType,
      inOppMap: latest != null,
      latestEdge: latestState?.edge ?? null,
      entryPriceEstimate: live.entryPriceEstimate,
      markPxFromLatest,
      markSource,
      effectiveMarkPx: markPxFromLatest,
    });

    const syntheticGrossEdge = markPxMtm != null ? 1 - markPxMtm : null;
    let computedMetrics = null as ReturnType<typeof applyDynamicMetricsToTrade> | null;
    if (dynCfg.engine) {
      const ref = getActiveTradeById(t.tradeId) ?? live;
      computedMetrics = applyDynamicMetricsToTrade(
        ref,
        latestState,
        syntheticGrossEdge,
        pol.feeBuffer,
        Date.now()
      );
    }

    const liveAfter = getActiveTradeById(t.tradeId) ?? live;
    const activeSt = toActiveState({
      ...liveAfter,
      grossEdgeAtEntry: liveAfter.grossEdgeAtEntry,
    });

    const exitOpts = {
      maxHoldingTimeMs: pol.maxHoldingTimeMs,
      stopLossPct: pol.stopLossPct,
      takeProfitPct: pol.takeProfitPct,
      edgeNormalizationThreshold: pol.edgeNormalizationThreshold,
      edgeCaptureDelta: pol.edgeCaptureDelta,
      edgeDeteriorationDelta: pol.edgeDeteriorationDelta,
    };

    const exitChain = {
      skipLegacyMaxHold: dynCfg.engine,
      skipLegacyEdgeCapture: dynCfg.engine && dynCfg.capturedEdge,
      skipLegacyEdgeDeterioration: dynCfg.engine && dynCfg.edgeDeteriorationFast,
    };

    let dynSnapshot: PaperExitDecisionSnapshot | null = null;
    const safetyExit = resolvePaperExitSafety(activeSt, latestState, exitOpts);
    let forcedExit = safetyExit;

    if (!forcedExit && dynCfg.engine && computedMetrics) {
      const ev = evaluateDynamicExit(liveAfter, computedMetrics, dynCfg);
      if (ev.shouldExit && ev.cause) {
        forcedExit = ev.cause;
        dynSnapshot = ev.decisionSnapshot;
      }
    }

    const shouldClose =
      forcedExit != null || shouldClosePaperTrade(activeSt, latestState, exitOpts, exitChain);

    if (shouldClose) {
      const exit: SimulatedExit = simulateExit(
        activeSt,
        latestState,
        exitOpts,
        exitChain,
        forcedExit
      );
      let mSnap = computedMetrics;
      if (mSnap == null) {
        const tmp = { ...liveAfter } as PaperTrade;
        mSnap = applyDynamicMetricsToTrade(tmp, latestState, syntheticGrossEdge, pol.feeBuffer, Date.now());
      }
      const exitDecisionSnapshot =
        dynSnapshot ?? buildExitDecisionSnapshot(exit.exitCondition, mSnap, dynCfg);
      const grossRealizedPnL = exit.realizedPnL;
      const fcClose = live.filledCapital;
      const estimatedEntryFees = fcClose * pol.feeBuffer;
      const estimatedExitFees = fcClose * pol.feeBuffer;
      const estimatedTotalFees = estimatedEntryFees + estimatedExitFees;
      const netRealizedPnL = grossRealizedPnL - estimatedTotalFees;
      const realizedReturnNet = fcClose > 1e-12 ? netRealizedPnL / fcClose : 0;
      const edgeAtExit = latestState != null ? latestState.edge : null;
      const exitPriceMarkSourceAtClose: "opp_map" | "mtm" | "fallback_no_latest" =
        latestState == null ? "fallback_no_latest" : markSource === "mtm" ? "mtm" : "opp_map";
      closeTrade(t.tradeId, {
        closedAt: exit.exitTimestamp,
        exitPriceEstimate: exit.exitPriceEstimate,
        edgeAtExit,
        exitPriceMarkSourceAtClose,
        exitConfidence: latest?.opp.confidence ?? t.entryConfidence,
        grossRealizedPnL,
        estimatedEntryFees,
        estimatedExitFees,
        estimatedTotalFees,
        netRealizedPnL,
        realizedPnL: netRealizedPnL,
        realizedReturn: realizedReturnNet,
        holdingTimeMs: Date.now() - new Date(t.openedAt).getTime(),
        maxAdverseExcursion: exit.maxAdverseExcursion,
        maxFavorableExcursion: exit.maxFavorableExcursion,
        exitCondition: exit.exitCondition,
        exitDecisionSnapshot,
      });
      recordGraphTradeClosed({
        ...t,
        status: "closed",
        grossRealizedPnL,
        estimatedEntryFees,
        estimatedExitFees,
        estimatedTotalFees,
        netRealizedPnL,
        realizedPnL: netRealizedPnL,
        closedAt: exit.exitTimestamp,
      });
      recordPaperEntryEconomicScoreAtClose(exit.exitCondition, t.entryEconomicScoreAtOpen);
      recordPaperExitProfileOutcome(entryProfileKey(t.sourceType, t.opportunityType), exit.exitCondition);
      if (exit.exitCondition === "no_progress_exit") {
        recordPaperOpenedThenNoProgressExit({
          tradeId: t.tradeId,
          profileKey: t.entryProfileKeyAtOpen ?? entryProfileKey(t.sourceType, t.opportunityType),
          entryEconomicScoreAtOpen: t.entryEconomicScoreAtOpen ?? null,
          progressProbabilityFactorAtOpen: t.progressProbabilityFactorAtOpen ?? null,
          realizedPnL: netRealizedPnL,
        });
      }
      recordPaperTradeLifecycleClose({
        tradeId: t.tradeId,
        opportunityId: t.opportunityId,
        opportunityType: t.opportunityType,
        inOppMap: latest != null,
        latestEdge: latestState?.edge ?? null,
        entryPriceEstimate: live.entryPriceEstimate,
        markPxFromLatest,
        markSource,
        effectiveMarkPx: markPxFromLatest,
        exitCondition: exit.exitCondition,
        exitPriceEstimate: exit.exitPriceEstimate,
        realizedPnL: netRealizedPnL,
        maxAdverseExcursion: exit.maxAdverseExcursion,
        maxFavorableExcursion: exit.maxFavorableExcursion,
        exitEqualsEntryBecauseNoLatest: latestState === null,
      });
      closed++;
    }
  }

  return { opened, closed, rejected };
}
