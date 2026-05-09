/**
 * Shared types for the Paper Trading + Capital Capacity Simulation layer.
 * Normalized opportunity shape used by capacity engine, execution simulator, and paper trade engine.
 */

import type { StructuralAssignmentReason } from "./graphStructuralMicroLane";

export type PaperSourceType = "standard" | "graph";

export type PaperOpportunityType =
  | "overround"
  | "underround"
  | "cross_market"
  | "graph_subset"
  | "graph_complement"
  | "graph_exclusive"
  | "graph_equivalence"
  | "graph_equivalence_micro"
  | "graph_subset_micro"
  | "graph_exclusive_micro"
  | "graph_cycle";

/** Proveniência da aresta no grafo (só `sourceType === graph`). */
export type PaperGraphDiagnosticProvenance =
  | "equivalent"
  | "subset"
  | "exclusive"
  | "complementary_strict"
  | "complementary_relaxed"
  | "cycle"
  | "unknown";

export interface NormalizedPaperOpportunity {
  opportunityId: string;
  sourceType: PaperSourceType;
  opportunityType: PaperOpportunityType;
  clusterId?: string;
  marketsInvolved: Array<{ marketId: string; question: string }>;
  edge: number;
  confidence: number;
  liquidity: number;
  spread: number;
  compositeScore?: number;
  rank?: number;
  /** Copiado de `GraphOpportunity.diagnosticRelationProvenance` na normalização. */
  graphDiagnosticProvenance?: PaperGraphDiagnosticProvenance;
  /** Só micro-lanes graph_*_micro; ver `graphStructuralMicroLane`. */
  structuralMicroLaneReason?: StructuralAssignmentReason;
}

export interface CapacityEstimate {
  opportunityId: string;
  estimatedGrossEdge: number;
  estimatedSlippageRate: number;
  estimatedImpactRate: number;
  estimatedNetEdge: number;
  maxDeployableCapital: number;
  recommendedCapital: number;
  capacityConfidence: number;
  reasoning: string;
}

export type PaperTradeStatus = "active" | "closed" | "rejected";

export type ExitCondition =
  | "edge_normalization"
  | "max_holding_time"
  | "stop_loss"
  | "take_profit"
  | "edge_capture"
  | "edge_deterioration"
  | "manual"
  | "profit_giveback"
  | "incremental_value_too_low"
  | "edge_fully_captured"
  | "edge_deteriorating_fast"
  | "capital_efficiency_exit"
  | "mtm_edge_divergence_exit"
  | "no_progress_exit"
  | "emergency_time_stop";

/** Inputs explícitos para auditar coerência edge vs MTM (opcional nas snapshots). */
export interface PaperExitDiagnosticInputs {
  markPx: number;
  markPxSource: "lastMarkPx" | "latestOpp" | "entry_fallback";
  /** grossEdge = 1 - markPx (alinhado ao preço usado em currentMarkedPnL). */
  grossEdgeFromMark: number;
  grossEdgeAtEntry: number;
  feeBuffer: number;
  capturedEdgeNumerator: number;
  capturedEdgeDenominator: number;
  remainingValueInputs: { currentNetEdge: number; filledCapital: number };
  mtmPnlInputs: { markPx: number; entryPriceEstimate: number; filledCapital: number };
  exitSignalConsistency: {
    currentNetEdgePositiveButPnlNegative: boolean;
    capturedEdgeZeroButPnlNonZero: boolean;
    remainingValueHighButDrawdownHigh: boolean;
  };
}

/** Persistido em trades fechados quando o motor dinâmico decide saída (ou legado + métricas finais). */
export interface PaperExitDecisionSnapshot {
  chosenExitCause: ExitCondition;
  timeInTradeMs: number;
  entryNetEdge: number;
  currentNetEdge: number;
  capturedEdgeRatio: number;
  edgeVelocity: number;
  expectedRemainingEdgeValue: number;
  bestMarkedPnL: number;
  currentMarkedPnL: number;
  drawdownFromPeakPnL: number;
  capitalEfficiencyScore: number;
  thresholdsAtDecision: Record<string, number | boolean | string>;
  diagnosticInputs?: PaperExitDiagnosticInputs;
  /** Preenchido quando a saída é `mtm_edge_divergence_exit`. */
  mtmEdgeDivergenceExit?: PaperMtmEdgeDivergenceExitSnapshot;
  /** Preenchido quando a saída é `no_progress_exit` (flat / sem evolução). */
  noProgressExit?: PaperNoProgressExitSnapshot;
}

/** Observabilidade da regra edge MTM vs PnL marcado. */
export interface PaperMtmEdgeDivergenceExitSnapshot {
  mtmEdgeDivergenceTriggered: true;
  thresholdsApplied: {
    minHoldBeforeMtmDivergenceExitMs: number;
    minNegativePnlForMtmDivergenceExitUsd: number;
    minDrawdownForMtmDivergenceExitUsd: number;
    requireRemainingHighForMtmDivergence: boolean;
  };
  valuesAtDecision: {
    timeInTradeMs: number;
    currentNetEdge: number;
    currentMarkedPnL: number;
    drawdownFromPeakPnL: number;
    expectedRemainingEdgeValue: number;
    exitSignalConsistency: PaperExitDiagnosticInputs["exitSignalConsistency"];
  };
}

/** Observabilidade: saída por trade plano / sem progresso económico. */
export interface PaperNoProgressExitSnapshot {
  noProgressTriggered: true;
  thresholdsApplied: {
    minHoldBeforeNoProgressExitMs: number;
    maxFlatPnlAbsUsd: number;
    maxCapturedEdgeRatioForNoProgressExit: number;
    maxFlatEdgeVelocityAbs: number;
    /** 0 = não aplicar tecto de drawdown; >0 exige drawdownFromPeakPnL <= valor. */
    noProgressMaxDrawdownFromPeakUsd: number;
    /** Não sair por no-progress se valor remanescente esperado > este USD (regime promissor). */
    skipNoProgressIfRemainingAboveUsd: number;
  };
  valuesAtDecision: {
    timeInTradeMs: number;
    currentMarkedPnL: number;
    capturedEdgeRatio: number;
    edgeVelocity: number;
    drawdownFromPeakPnL: number;
    expectedRemainingEdgeValue: number;
  };
}

export interface PaperTrade {
  tradeId: string;
  opportunityId: string;
  sourceType: PaperSourceType;
  opportunityType: PaperOpportunityType;
  clusterId?: string;
  marketsInvolved: Array<{ marketId: string; question: string }>;
  openedAt: string;
  closedAt: string | null;
  status: PaperTradeStatus;
  rejectionReason?: string;
  grossEdgeAtEntry: number;
  netEdgeAtEntry: number;
  recommendedCapital: number;
  requestedCapital: number;
  filledCapital: number;
  entryPriceEstimate: number;
  exitPriceEstimate?: number;
  /**
   * Instrumentação (fecho): `latestState.edge` usado em `simulateExit` (`exitPrice = 1 - edge`) quando houve `latestState`.
   * `null` persistido = `latestState === null` (fallback `exitPrice = entry`). Ausente = trade legada.
   */
  edgeAtExit?: number | null;
  /**
   * Instrumentação (fecho): origem do `latestState` no ciclo do `paperTradeEngine` (`opp_map` | `mtm` | `fallback_no_latest`).
   */
  exitPriceMarkSourceAtClose?: "opp_map" | "mtm" | "fallback_no_latest";
  entryConfidence: number;
  exitConfidence?: number;
  /**
   * PnL **líquido** USD (fechamentos novos, após taxas estimadas).
   * Trades legados: era só o **bruto** do simulador; usar `getClosedTradeNetRealizedPnL` para líquido inferido.
   */
  realizedPnL: number;
  /** Bruto do `simulateExit`: `filledCapital * (exitPrice − entryPrice) / entryPrice`. Opcional em legados. */
  grossRealizedPnL?: number;
  /** Taxa estimada por perna na entrada: `filledCapital × feeBuffer`. */
  estimatedEntryFees?: number;
  /** Taxa estimada por perna na saída (mesmo modelo que entrada económica). */
  estimatedExitFees?: number;
  estimatedTotalFees?: number;
  /** `grossRealizedPnL − estimatedTotalFees`; espelhado em `realizedPnL` nos fechos novos. */
  netRealizedPnL?: number;
  /** Retorno: `netRealizedPnL / filledCapital` (novos); legado = bruto/filled. */
  realizedReturn: number;
  holdingTimeMs: number;
  maxAdverseExcursion: number;
  maxFavorableExcursion: number;
  exitCondition?: ExitCondition;
  notes?: string;
  /** Último mark MTM (ciclo paper); opcional */
  lastMarkPx?: number;
  lastMarkAt?: string;
  /** Estado económico dinâmico (opcional; preenchido quando o motor dinâmico está activo). */
  currentNetEdge?: number;
  lastNetEdge?: number;
  bestMarkedPnL?: number;
  worstMarkedPnL?: number;
  bestMarkedAt?: string;
  capturedEdgeRatio?: number;
  edgeVelocity?: number;
  expectedRemainingEdgeValue?: number;
  capitalEfficiencyScore?: number;
  drawdownFromPeakPnL?: number;
  lastDynamicTickAt?: number;
  exitDecisionSnapshot?: PaperExitDecisionSnapshot;
  /** Diagnóstico de entrada (opcional). */
  entryEconomicScoreAtOpen?: number;
  progressProbabilityFactorAtOpen?: number;
  entryProfileKeyAtOpen?: string;
  /** Preservado na abertura para métricas downstream por proveniência. */
  graphDiagnosticProvenanceAtOpen?: PaperGraphDiagnosticProvenance;
  /** Só graph micro-lanes; razão estrutural da classificação no scan. */
  structuralMicroLaneReasonAtOpen?: StructuralAssignmentReason;
}
