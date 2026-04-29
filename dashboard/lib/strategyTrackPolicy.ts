/**
 * Modo single-track: uma única faixa activa promovida; outros experimentos arquivados (sem nova classe de estratégia).
 */

import type { LiveExperimentId } from "./liveExperimentStatsShared";

export const ACTIVE_SINGLE_TRACK_EXPERIMENT_ID: LiveExperimentId = "cross_venue_anchor_1823789";

export const ACTIVE_SINGLE_TRACK_MARKET_ID = "1823789" as const;

/** Experimentos fechados para promoção; workers podem continuar a correr só para arquivo operacional. */
export const ARCHIVED_CONSISTENTLY_NEGATIVE_EXPERIMENT_IDS: ReadonlySet<LiveExperimentId> = new Set<
  LiveExperimentId
>(["final_neg_risk_31552", "reach_april_btc_reaction_monitor"]);

export type StrategyTrackClassification =
  | "approved_for_controlled_paper_phase"
  | "active_single_track"
  | "archived_consistently_negative";

export const STRATEGY_TRACK_MODE_SINGLE = "single_track_1823789" as const;

export function getStrategyTrackClassification(
  experimentId: LiveExperimentId,
): StrategyTrackClassification {
  if (experimentId === ACTIVE_SINGLE_TRACK_EXPERIMENT_ID) {
    return "approved_for_controlled_paper_phase";
  }
  if (ARCHIVED_CONSISTENTLY_NEGATIVE_EXPERIMENT_IDS.has(experimentId)) {
    return "archived_consistently_negative";
  }
  return "archived_consistently_negative";
}

export function isActiveSingleTrackExperiment(experimentId: LiveExperimentId): boolean {
  return experimentId === ACTIVE_SINGLE_TRACK_EXPERIMENT_ID;
}

export function isPromotionActiveTrackClassification(
  classification: StrategyTrackClassification,
): boolean {
  return (
    classification === "approved_for_controlled_paper_phase" ||
    classification === "active_single_track"
  );
}
