/**
 * Execution / sizing gate — apenas mercado 1823789 (faixa single-track).
 * Baseline: mesmo snapshot lógico que narrow-validation (custos refinados); aqui só realism de execução e sensibilidade a tamanho.
 * Sem nova classe de estratégia, sem expansão de universo.
 */

import { fetchGammaMarketRawJson, fetchParsedClobBook, resolveYesClobTokenId } from "./clobMicrostructure";
import { buildCrossVenueAnchor1823789NarrowValidationDigest } from "./crossVenueAnchor1823789NarrowValidation";
import {
  ACTIVE_SINGLE_TRACK_EXPERIMENT_ID,
  ACTIVE_SINGLE_TRACK_MARKET_ID,
  getStrategyTrackClassification,
  STRATEGY_TRACK_MODE_SINGLE,
  type StrategyTrackClassification,
} from "./strategyTrackPolicy";

export const EXECUTION_GATE_PROBE_VERSION = "cross-venue-anchor-1823789-execution-gate-v1" as const;

export type ExecutionSizeTierKey = "1x" | "2x" | "5x";

export type ExecutionGateVerdict =
  | "blocked_by_execution"
  | "fragile_but_paperable"
  | "survives_execution_gate"
  | "survives_and_scalable";

export type ExecutionFragilityVerdict = "robust" | "moderate" | "fragile";

export type ExecutionPromotionVerdict = "blocked" | "paper_only" | "limited_live" | "scalable_live";

export type FillPlausibility = "high" | "medium" | "low";

export interface CrossVenueAnchor1823789ExecutionGateDigest {
  probeVersion: typeof EXECUTION_GATE_PROBE_VERSION;
  readDisclaimer: string;
  strategyTrackMode: typeof STRATEGY_TRACK_MODE_SINGLE;
  strategyTrackClassification: StrategyTrackClassification;
  experimentId: typeof ACTIVE_SINGLE_TRACK_EXPERIMENT_ID;
  narrowValidationVerdict: string;
  executionGateVerdict: ExecutionGateVerdict;
  baselineNet: number;
  minNetAfterExecutionAdjustments: number;
  bestTier: ExecutionSizeTierKey;
  worstTier: ExecutionSizeTierKey;
  executionGateSummaryLine: string;
  marketId: typeof ACTIVE_SINGLE_TRACK_MARKET_ID;
  marketTitle: string;
  baselineObservationIso: string;
  rawAnchorGap: number;
  refinedFairValue: number;
  anchorPriceObserved: number;
  polymarketPriceObserved: number;
  /** Custos baseline do narrow (mesmo snapshot); útil para paper execution sem segunda chamada. */
  baselineCosts: {
    estimatedEntryCost: number;
    estimatedExitCost: number;
    estimatedHedgeCost: number;
    estimatedLegRiskCost: number;
  };
  clobBestBid: number;
  clobBestAsk: number;
  clobMid: number;
  clobSpreadUsed: number;
  clobBidLevels: number;
  clobAskLevels: number;
  entryDepthAvailable: number;
  exitDepthAvailable: number;
  estimatedEntrySlippageByTier: Record<ExecutionSizeTierKey, number>;
  estimatedExitSlippageByTier: Record<ExecutionSizeTierKey, number>;
  fillPlausibility: FillPlausibility;
  adverseMoveExecutionPenalty: number;
  estimatedNetAnchorCycleByTier: Record<ExecutionSizeTierKey, number>;
  executionFragilityVerdict: ExecutionFragilityVerdict;
  executionPromotionVerdict: ExecutionPromotionVerdict;
  microstructureObservationState:
    | "valid_book_two_sided"
    | "valid_book_one_sided"
    | "clob_book_unavailable";
  microstructureDetail: string;
  clobBookStructure: "two_sided" | "one_sided" | "none";
  supportingNote: string;
  computedAt: string;
  /** Token id CLOB outcome YES (Gamma) — intent shadow/live. */
  yesOutcomeTokenId: string;
}

function r6(n: number): number {
  return Math.round(n * 1_000_000) / 1_000_000;
}

/** Unidade de referência em contratos YES (top-of-book depth proxy); tiers escala a procura vs profundidade. */
export const REF_UNIT_SHARES = 400;

function slippageFromUtilization(spread: number, utilization: number): number {
  if (!(utilization >= 0) || !Number.isFinite(spread)) return 0;
  const u = Math.min(2.5, utilization);
  return r6(Math.min(0.06, spread * (0.35 * u + 0.12 * u * u)));
}

export async function buildCrossVenueAnchor1823789ExecutionGateDigest(): Promise<CrossVenueAnchor1823789ExecutionGateDigest> {
  const narrow = await buildCrossVenueAnchor1823789NarrowValidationDigest();
  const baseline = narrow.stressScenarios.find(s => s.label === "baseline");
  const baselineNet = baseline?.estimatedNetAnchorCycle ?? r6(narrow.refinedFairValue - narrow.polymarketPriceObserved);

  const raw = await fetchGammaMarketRawJson(ACTIVE_SINGLE_TRACK_MARKET_ID);
  const yesToken = raw ? resolveYesClobTokenId(raw) : "";
  const book = yesToken ? await fetchParsedClobBook(yesToken) : null;

  const entryDepthAvailable = book?.depthAskTop3 ?? 0;
  const exitDepthAvailable = book?.depthBidTop3 ?? 0;
  const spread = book?.spread ?? 0;
  const clobBestBid = book?.bestBid ?? 0;
  const clobBestAsk = book?.bestAsk ?? 0;
  const clobMid = book?.mid ?? 0;
  const clobBidLevels = book?.bidLevels ?? 0;
  const clobAskLevels = book?.askLevels ?? 0;
  const clobBookStructure: "two_sided" | "one_sided" | "none" =
    book && clobBidLevels > 0 && clobAskLevels > 0 ? "two_sided" : book ? "one_sided" : "none";
  const microstructureObservationState: CrossVenueAnchor1823789ExecutionGateDigest["microstructureObservationState"] =
    clobBookStructure === "two_sided"
      ? "valid_book_two_sided"
      : clobBookStructure === "one_sided"
        ? "valid_book_one_sided"
        : "clob_book_unavailable";
  const microstructureDetail =
    clobBookStructure === "none"
      ? "book_fetch_or_parse_failed"
      : clobBidLevels > 0 && clobAskLevels > 0
        ? `book_levels bid=${clobBidLevels} ask=${clobAskLevels}`
        : clobAskLevels > 0 && clobBidLevels === 0
          ? `book_levels bid=${clobBidLevels} ask=${clobAskLevels}|asks_only`
          : clobBidLevels > 0 && clobAskLevels === 0
            ? `book_levels bid=${clobBidLevels} ask=${clobAskLevels}|bids_only`
            : `book_levels bid=${clobBidLevels} ask=${clobAskLevels}`;

  const tiers: ExecutionSizeTierKey[] = ["1x", "2x", "5x"];
  const tierMult: Record<ExecutionSizeTierKey, number> = { "1x": 1, "2x": 2, "5x": 5 };

  const estimatedEntrySlippageByTier: Record<ExecutionSizeTierKey, number> = {
    "1x": 0,
    "2x": 0,
    "5x": 0,
  };
  const estimatedExitSlippageByTier: Record<ExecutionSizeTierKey, number> = {
    "1x": 0,
    "2x": 0,
    "5x": 0,
  };
  const estimatedNetAnchorCycleByTier: Record<ExecutionSizeTierKey, number> = {
    "1x": 0,
    "2x": 0,
    "5x": 0,
  };

  const minDepth = Math.min(entryDepthAvailable, exitDepthAvailable);
  const demand5x = REF_UNIT_SHARES * 5;
  let fillPlausibility: FillPlausibility = "low";
  if (minDepth >= demand5x * 0.35) fillPlausibility = "high";
  else if (minDepth >= demand5x * 0.12) fillPlausibility = "medium";
  else fillPlausibility = "low";

  let adverseMoveExecutionPenalty = 0;

  for (const tk of tiers) {
    const m = tierMult[tk];
    const demand = REF_UNIT_SHARES * m;
    const uEntry = demand / Math.max(entryDepthAvailable, 1e-6);
    const uExit = demand / Math.max(exitDepthAvailable, 1e-6);
    const entrySlip = slippageFromUtilization(spread, uEntry);
    const exitSlip = slippageFromUtilization(spread, uExit);
    estimatedEntrySlippageByTier[tk] = entrySlip;
    estimatedExitSlippageByTier[tk] = exitSlip;

    // Movimento adverso durante a janela: ordem de grandeza ancorada ao spread do YES (não escalares fixas grandes).
    const adverse = r6(spread * (0.12 * m + 0.08 * (m - 1)) + 0.00006 * m * m);
    if (tk === "5x") adverseMoveExecutionPenalty = adverse;

    const hedgeExecutionBurden = r6(narrow.baselineCosts.estimatedHedgeCost * 0.12 * (m - 1));

    const executionDrag = r6(entrySlip + exitSlip + adverse + hedgeExecutionBurden);
    estimatedNetAnchorCycleByTier[tk] = r6(baselineNet - executionDrag);
  }

  const netVals = tiers.map(t => estimatedNetAnchorCycleByTier[t]);
  const minNetAfterExecutionAdjustments = r6(Math.min(...netVals));
  const bestTier = tiers.reduce((a, b) =>
    estimatedNetAnchorCycleByTier[a] >= estimatedNetAnchorCycleByTier[b] ? a : b,
  );
  const worstTier = tiers.reduce((a, b) =>
    estimatedNetAnchorCycleByTier[a] <= estimatedNetAnchorCycleByTier[b] ? a : b,
  );

  let executionGateVerdict: ExecutionGateVerdict = "survives_execution_gate";
  if (minNetAfterExecutionAdjustments <= 0 || baselineNet <= 0) {
    executionGateVerdict = "blocked_by_execution";
  } else if (
    estimatedNetAnchorCycleByTier["5x"] <= 0 ||
    estimatedNetAnchorCycleByTier["5x"] < 0.003 ||
    (estimatedNetAnchorCycleByTier["5x"] < estimatedNetAnchorCycleByTier["1x"] * 0.45 && fillPlausibility === "low")
  ) {
    executionGateVerdict = "fragile_but_paperable";
  } else if (
    minNetAfterExecutionAdjustments >= 0.006 &&
    estimatedNetAnchorCycleByTier["5x"] >= 0.006 &&
    fillPlausibility !== "low"
  ) {
    executionGateVerdict = "survives_and_scalable";
  } else {
    executionGateVerdict = "survives_execution_gate";
  }

  if (narrow.narrowValidationVerdict !== "survives_stress_suite") {
    executionGateVerdict = "blocked_by_execution";
  }

  let executionFragilityVerdict: ExecutionFragilityVerdict = "moderate";
  if (executionGateVerdict === "blocked_by_execution") executionFragilityVerdict = "fragile";
  else if (executionGateVerdict === "survives_and_scalable") executionFragilityVerdict = "robust";
  else if (executionGateVerdict === "fragile_but_paperable") executionFragilityVerdict = "fragile";

  let executionPromotionVerdict: ExecutionPromotionVerdict = "limited_live";
  if (executionGateVerdict === "blocked_by_execution") executionPromotionVerdict = "blocked";
  else if (executionGateVerdict === "fragile_but_paperable") executionPromotionVerdict = "paper_only";
  else if (executionGateVerdict === "survives_execution_gate") executionPromotionVerdict = "limited_live";
  else if (executionGateVerdict === "survives_and_scalable") executionPromotionVerdict = "scalable_live";

  const supportingNote = [
    `ref_unit_shares=${REF_UNIT_SHARES}`,
    `clob_spread=${r6(spread)}`,
    `depth_ask_top3=${r6(entryDepthAvailable)}`,
    `depth_bid_top3=${r6(exitDepthAvailable)}`,
    `narrow_baseline_net=${r6(baselineNet)}`,
    `narrow_verdict=${narrow.narrowValidationVerdict}`,
    `fill_plausibility=${fillPlausibility}`,
  ].join("|");

  const executionGateSummaryLine = `exec_gate_1823789_v1 verdict=${executionGateVerdict} baseline_net=${baselineNet.toFixed(6)} min_net_adj=${minNetAfterExecutionAdjustments.toFixed(6)} best=${bestTier} worst=${worstTier} fill=${fillPlausibility}`;

  return {
    probeVersion: EXECUTION_GATE_PROBE_VERSION,
    readDisclaimer:
      "Execution/sizing gate for single-track market 1823789 only. Baseline net from narrow-validation stack; adjustments are paper proxies for depth, tiered slippage, fill plausibility, adverse move during execution, and incremental hedge execution burden. Not live orders; no universe expansion; same strategy class.",
    strategyTrackMode: STRATEGY_TRACK_MODE_SINGLE,
    strategyTrackClassification: getStrategyTrackClassification(
      ACTIVE_SINGLE_TRACK_EXPERIMENT_ID,
    ),
    experimentId: ACTIVE_SINGLE_TRACK_EXPERIMENT_ID,
    narrowValidationVerdict: narrow.narrowValidationVerdict,
    executionGateVerdict,
    baselineNet: r6(baselineNet),
    minNetAfterExecutionAdjustments,
    bestTier,
    worstTier,
    executionGateSummaryLine,
    marketId: ACTIVE_SINGLE_TRACK_MARKET_ID,
    marketTitle: narrow.marketTitle,
    baselineObservationIso: narrow.baselineObservationIso,
    rawAnchorGap: r6(narrow.polymarketPriceObserved - narrow.refinedFairValue),
    refinedFairValue: narrow.refinedFairValue,
    anchorPriceObserved: narrow.anchorPriceObserved,
    polymarketPriceObserved: narrow.polymarketPriceObserved,
    baselineCosts: { ...narrow.baselineCosts },
    clobBestBid: r6(clobBestBid),
    clobBestAsk: r6(clobBestAsk),
    clobMid: r6(clobMid),
    clobSpreadUsed: r6(spread),
    clobBidLevels,
    clobAskLevels,
    entryDepthAvailable: r6(entryDepthAvailable),
    exitDepthAvailable: r6(exitDepthAvailable),
    estimatedEntrySlippageByTier,
    estimatedExitSlippageByTier,
    fillPlausibility,
    adverseMoveExecutionPenalty,
    estimatedNetAnchorCycleByTier,
    executionFragilityVerdict,
    executionPromotionVerdict,
    microstructureObservationState,
    microstructureDetail,
    clobBookStructure,
    supportingNote,
    computedAt: new Date().toISOString(),
    yesOutcomeTokenId: yesToken,
  };
}
