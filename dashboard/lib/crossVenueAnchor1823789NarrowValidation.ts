/**
 * Validação estreita só para mercado 1823789 (faixa activa): paper mais conservador + choques de spread / hedge / leg risk.
 * Sem expansão de universo; não altera a classe de estratégia subjacente (reutiliza observação refinada existente).
 */

import { buildCrossVenueAnchor1823789RefinedObservation } from "./crossVenueAnchorPilotRefined";
import {
  ACTIVE_SINGLE_TRACK_EXPERIMENT_ID,
  ACTIVE_SINGLE_TRACK_MARKET_ID,
  getStrategyTrackClassification,
  STRATEGY_TRACK_MODE_SINGLE,
  type StrategyTrackClassification,
} from "./strategyTrackPolicy";

export type NarrowStressScenarioLabel =
  | "baseline"
  | "spread_shock_plus_50pct_roundtrip"
  | "spread_shock_plus_100pct_roundtrip"
  | "hedge_shock_plus_50pct"
  | "leg_shock_plus_50pct"
  | "combined_mild_shock";

export type NarrowValidationVerdict =
  | "survives_stress_suite"
  | "fragile_under_stress"
  | "not_viable_under_stress";

export interface NarrowStressRow {
  label: NarrowStressScenarioLabel;
  spreadRoundtripCostMultiplier: number;
  hedgeCostMultiplier: number;
  legRiskMultiplier: number;
  estimatedNetAnchorCycle: number;
  supportingNote: string;
}

export interface CrossVenueAnchor1823789NarrowValidationDigest {
  probeVersion: "cross-venue-anchor-1823789-narrow-validation-v1";
  readDisclaimer: string;
  strategyTrackMode: typeof STRATEGY_TRACK_MODE_SINGLE;
  strategyTrackClassification: StrategyTrackClassification;
  marketId: typeof ACTIVE_SINGLE_TRACK_MARKET_ID;
  marketTitle: string;
  experimentId: typeof ACTIVE_SINGLE_TRACK_EXPERIMENT_ID;
  baselineObservationIso: string;
  anchorPriceObserved: number;
  refinedFairValue: number;
  polymarketPriceObserved: number;
  impliedEdgeVersusFair: number;
  /** Custos observados no snapshot (entrada+saída+hedge+leg). */
  baselineCosts: {
    estimatedEntryCost: number;
    estimatedExitCost: number;
    estimatedHedgeCost: number;
    estimatedLegRiskCost: number;
  };
  stressScenarios: NarrowStressRow[];
  narrowValidationVerdict: NarrowValidationVerdict;
  narrowValidationSummaryLine: string;
  computedAt: string;
}

function r6(n: number): number {
  return Math.round(n * 1_000_000) / 1_000_000;
}

function netUnderStress(
  edgeMag: number,
  entry: number,
  exit: number,
  hedge: number,
  leg: number,
  spreadMult: number,
  hedgeMult: number,
  legMult: number,
): number {
  return r6(edgeMag - entry * spreadMult - exit * spreadMult - hedge * hedgeMult - leg * legMult);
}

export async function buildCrossVenueAnchor1823789NarrowValidationDigest(): Promise<CrossVenueAnchor1823789NarrowValidationDigest> {
  const obs = await buildCrossVenueAnchor1823789RefinedObservation();
  const row = obs.row;

  const edgeMag = r6(row.refinedFairValue - row.polymarketPriceObserved);
  const entry = row.estimatedEntryCost;
  const exit = row.estimatedExitCost;
  const hedge = row.estimatedHedgeCost;
  const leg = row.estimatedLegRiskCost;

  const scenarios: NarrowStressRow[] = [
    {
      label: "baseline",
      spreadRoundtripCostMultiplier: 1,
      hedgeCostMultiplier: 1,
      legRiskMultiplier: 1,
      estimatedNetAnchorCycle: row.estimatedNetAnchorCycle,
      supportingNote: "uses_worker_refined_row_net",
    },
    {
      label: "spread_shock_plus_50pct_roundtrip",
      spreadRoundtripCostMultiplier: 1.5,
      hedgeCostMultiplier: 1,
      legRiskMultiplier: 1,
      estimatedNetAnchorCycle: netUnderStress(edgeMag, entry, exit, hedge, leg, 1.5, 1, 1),
      supportingNote: "entry_exit_scaled_pm_microstructure_stress",
    },
    {
      label: "spread_shock_plus_100pct_roundtrip",
      spreadRoundtripCostMultiplier: 2,
      hedgeCostMultiplier: 1,
      legRiskMultiplier: 1,
      estimatedNetAnchorCycle: netUnderStress(edgeMag, entry, exit, hedge, leg, 2, 1, 1),
      supportingNote: "entry_exit_doubled_proxy_worse_fill_path",
    },
    {
      label: "hedge_shock_plus_50pct",
      spreadRoundtripCostMultiplier: 1,
      hedgeCostMultiplier: 1.5,
      legRiskMultiplier: 1,
      estimatedNetAnchorCycle: netUnderStress(edgeMag, entry, exit, hedge, leg, 1, 1.5, 1),
      supportingNote: "venue_hedge_slippage_proxy",
    },
    {
      label: "leg_shock_plus_50pct",
      spreadRoundtripCostMultiplier: 1,
      hedgeCostMultiplier: 1,
      legRiskMultiplier: 1.5,
      estimatedNetAnchorCycle: netUnderStress(edgeMag, entry, exit, hedge, leg, 1, 1, 1.5),
      supportingNote: "inventory_timing_leg_risk_proxy",
    },
    {
      label: "combined_mild_shock",
      spreadRoundtripCostMultiplier: 1.35,
      hedgeCostMultiplier: 1.35,
      legRiskMultiplier: 1.35,
      estimatedNetAnchorCycle: netUnderStress(edgeMag, entry, exit, hedge, leg, 1.35, 1.35, 1.35),
      supportingNote: "joint_micro_hedge_leg_stress",
    },
  ];

  const baselineNet = scenarios[0].estimatedNetAnchorCycle;
  const combined = scenarios.find(s => s.label === "combined_mild_shock")!.estimatedNetAnchorCycle;
  const minNet = Math.min(...scenarios.map(s => s.estimatedNetAnchorCycle));

  let narrowValidationVerdict: NarrowValidationVerdict = "fragile_under_stress";
  if (baselineNet <= 0) {
    narrowValidationVerdict = "not_viable_under_stress";
  } else if (combined > 0 && minNet > -0.004) {
    narrowValidationVerdict = "survives_stress_suite";
  } else if (combined <= 0 || minNet <= -0.01) {
    narrowValidationVerdict = "not_viable_under_stress";
  } else {
    narrowValidationVerdict = "fragile_under_stress";
  }

  const narrowValidationSummaryLine = `narrow_1823789_v1 verdict=${narrowValidationVerdict} baseline_net=${baselineNet.toFixed(6)} combined_mild_net=${combined.toFixed(6)} min_net=${minNet.toFixed(6)}`;

  return {
    probeVersion: "cross-venue-anchor-1823789-narrow-validation-v1",
    readDisclaimer:
      "Narrow validation for promoted single-track market 1823789 only. Reuses one refined observation snapshot; stress applies multipliers to roundtrip PM microstructure costs, hedge, and leg-risk proxies — not a new strategy class and no universe expansion. Paper-only stress; not execution.",
    strategyTrackMode: STRATEGY_TRACK_MODE_SINGLE,
    strategyTrackClassification: getStrategyTrackClassification(
      ACTIVE_SINGLE_TRACK_EXPERIMENT_ID,
    ),
    marketId: ACTIVE_SINGLE_TRACK_MARKET_ID,
    marketTitle: row.marketTitle,
    experimentId: ACTIVE_SINGLE_TRACK_EXPERIMENT_ID,
    baselineObservationIso: obs.isoTimestamp,
    anchorPriceObserved: row.anchorPriceObserved,
    refinedFairValue: row.refinedFairValue,
    polymarketPriceObserved: row.polymarketPriceObserved,
    impliedEdgeVersusFair: edgeMag,
    baselineCosts: {
      estimatedEntryCost: row.estimatedEntryCost,
      estimatedExitCost: row.estimatedExitCost,
      estimatedHedgeCost: row.estimatedHedgeCost,
      estimatedLegRiskCost: row.estimatedLegRiskCost,
    },
    stressScenarios: scenarios,
    narrowValidationVerdict,
    narrowValidationSummaryLine,
    computedAt: new Date().toISOString(),
  };
}
