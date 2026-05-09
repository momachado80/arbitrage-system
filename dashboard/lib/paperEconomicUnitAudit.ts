/**
 * Auditoria só de leitura: unidades e fórmula de PnL paper (executionSimulator).
 * Não altera política nem cálculos em runtime.
 */

import {
  effectiveGraphProvenanceForClosedAnalytics,
  PAPER_GRAPH_PROVENANCE_KEYS,
} from "./graphOpportunityPaperImpact";
import { getPaperPortfolio } from "./paperPortfolioStore";
import { isClosedTradeWithFiniteRealizedPnl } from "./paperClosedTradesMetrics";
import {
  getClosedTradeEstimatedTotalFees,
  getClosedTradeGrossRealizedPnL,
  getClosedTradeNetRealizedPnL,
} from "./paperRealizedPnlSemantics";
import { resolvePaperPolicyFromEnv } from "./paperTradeEngine";
import type { PaperGraphDiagnosticProvenance, PaperTrade } from "./paperTypes";

const SAMPLE_CAP = 10;
const RECON_EPS_ABS = 0.02;
const RECON_EPS_REL = 1e-5;

export type ClosedTradeEconomicSample = {
  tradeId: string;
  graphDiagnosticProvenanceAtOpen: PaperGraphDiagnosticProvenance | null;
  effectiveProvenance: string;
  filledCapital: number;
  entryPriceEstimate: number | null;
  exitPriceEstimate: number | null;
  pnlPctFromPrices: number | null;
  priceDelta: number | null;
  /** Líquido (KPI): persistido ou inferido com taxas. */
  netRealizedPnL: number;
  grossRealizedPnL: number;
  estimatedTotalFees: number;
  /** Valor bruto do campo `realizedPnL` no registo (nos fechos novos = líquido). */
  realizedPnLFieldStored: number | null;
  realizedReturnStored: number;
  /** Texto fixo alinhado a `simulateExit` em executionSimulator.ts */
  formulaReconstruction: string;
  reconstructedRealizedPnL: number | null;
  reconstructionAbsDiff: number | null;
  reconstructionMatches: boolean | null;
  /** filledCapital / entryPrice — “acções” sintéticas ao preço de entrada. */
  imputedQtySharesApprox: number | null;
  ratioRealizedPnlToFilledCapital: number | null;
  warningFlags: string[];
};

export type ProvenanceEconomicUnitAgg = {
  provenance: string;
  tradeCount: number;
  avgRealizedPnlToFilledCapitalRatio: number | null;
  medianRealizedPnlToFilledCapitalRatio: number | null;
  maxRealizedPnlToFilledCapitalRatio: number | null;
  countTradesWhereAbsRealizedPnlExceedsFilledCapital: number;
};

export type PaperEconomicUnitAudit = {
  computedAt: string;
  note: string;
  realizedPnlFormulaSummary: string;
  filledCapitalFormulaSummary: string;
  entryExitPriceConventionSummary: string;
  implicitQuantityAndUsdSummary: string;
  feesInSimulatedRealizedPnlSummary: string;
  sampleClosedTradeEconomicAudit: ClosedTradeEconomicSample[];
  aggregatesByProvenance: ProvenanceEconomicUnitAgg[];
  totals: {
    closedTradesWithFinitePnlAnalyzed: number;
    countReconstructionMismatch: number;
    countMissingExitPrice: number;
  };
};

function round4(n: number): number {
  return Math.round(n * 10000) / 10000;
}

function medianSorted(sorted: number[]): number | null {
  if (sorted.length === 0) return null;
  const m = [...sorted].sort((a, b) => a - b);
  const mid = Math.floor(m.length / 2);
  return m.length % 2 === 1 ? m[mid]! : (m[mid - 1]! + m[mid]!) / 2;
}

function provenanceBucket(t: PaperTrade): string {
  if (t.sourceType !== "graph") return "non_graph";
  const p = effectiveGraphProvenanceForClosedAnalytics(t);
  return p ?? "unknown";
}

function reconstructExitPnL(trade: PaperTrade): {
  pnlPct: number | null;
  priceDelta: number | null;
  reconstructed: number | null;
  imputedQty: number | null;
  formula: string;
} {
  const ep = trade.entryPriceEstimate;
  const xp = trade.exitPriceEstimate;
  const fc = trade.filledCapital;
  const baseFormula =
    "realizedPnL = filledCapital * ((exitPriceEstimate - entryPriceEstimate) / entryPriceEstimate); entryPriceEstimate = 1 - edge_at_entry; exitPriceEstimate = 1 - edge_at_exit (ou entry se sem latest opp no simulateExit)";
  if (typeof ep !== "number" || !Number.isFinite(ep) || ep <= 0) {
    return { pnlPct: null, priceDelta: null, reconstructed: null, imputedQty: null, formula: baseFormula };
  }
  if (typeof xp !== "number" || !Number.isFinite(xp)) {
    return {
      pnlPct: null,
      priceDelta: null,
      reconstructed: null,
      imputedQty: round4(fc / ep),
      formula: baseFormula,
    };
  }
  const pnlPct = (xp - ep) / ep;
  const reconstructed = fc * pnlPct;
  const imputedQty = fc / ep;
  return {
    pnlPct: round4(pnlPct),
    priceDelta: round4(xp - ep),
    reconstructed: round4(reconstructed),
    imputedQty: round4(imputedQty),
    formula: baseFormula,
  };
}

function buildSample(t: PaperTrade, feeBufferPerLeg: number): ClosedTradeEconomicSample {
  const r = reconstructExitPnL(t);
  const grossStored = getClosedTradeGrossRealizedPnL(t);
  const netEff = getClosedTradeNetRealizedPnL(t, feeBufferPerLeg);
  const feesEff = getClosedTradeEstimatedTotalFees(t, feeBufferPerLeg);
  const fieldStored =
    typeof t.realizedPnL === "number" && Number.isFinite(t.realizedPnL) ? t.realizedPnL : null;
  let reconstructionMatches: boolean | null = null;
  let reconstructionAbsDiff: number | null = null;
  if (r.reconstructed != null && Number.isFinite(grossStored)) {
    reconstructionAbsDiff = round4(Math.abs(r.reconstructed - grossStored));
    const tol = Math.max(RECON_EPS_ABS, Math.abs(grossStored) * RECON_EPS_REL);
    reconstructionMatches = reconstructionAbsDiff <= tol;
  }

  const flags: string[] = [];
  const fc = t.filledCapital;
  if (t.exitPriceEstimate == null || typeof t.exitPriceEstimate !== "number") {
    flags.push("missing_exit_price");
  }
  if (typeof fc === "number" && fc > 0 && Math.abs(grossStored) > fc + 1e-9) {
    flags.push("abs_gross_pnl_gt_filled_capital");
  }
  if (reconstructionMatches === false) {
    flags.push("reconstruction_mismatch_vs_gross_fields");
  }
  if (
    typeof fc === "number" &&
    fc > 0 &&
    typeof r.reconstructed === "number" &&
    Math.abs(r.reconstructed) > fc * 10
  ) {
    flags.push("extreme_return_vs_capital_sanity_check");
  }

  const ratio =
    typeof fc === "number" && fc > 0 && Number.isFinite(netEff) ? round4(netEff / fc) : null;

  const provOpen = t.graphDiagnosticProvenanceAtOpen ?? null;

  return {
    tradeId: t.tradeId,
    graphDiagnosticProvenanceAtOpen: provOpen,
    effectiveProvenance: provenanceBucket(t),
    filledCapital: round4(fc),
    entryPriceEstimate:
      typeof t.entryPriceEstimate === "number" && Number.isFinite(t.entryPriceEstimate)
        ? round4(t.entryPriceEstimate)
        : null,
    exitPriceEstimate:
      typeof t.exitPriceEstimate === "number" && Number.isFinite(t.exitPriceEstimate)
        ? round4(t.exitPriceEstimate)
        : null,
    pnlPctFromPrices: r.pnlPct,
    priceDelta: r.priceDelta,
    netRealizedPnL: round4(netEff),
    grossRealizedPnL: round4(grossStored),
    estimatedTotalFees: round4(feesEff),
    realizedPnLFieldStored: fieldStored != null ? round4(fieldStored) : null,
    realizedReturnStored: round4(t.realizedReturn),
    formulaReconstruction: r.formula,
    reconstructedRealizedPnL: r.reconstructed,
    reconstructionAbsDiff,
    reconstructionMatches,
    imputedQtySharesApprox: r.imputedQty,
    ratioRealizedPnlToFilledCapital: ratio,
    warningFlags: flags,
  };
}

export function buildPaperEconomicUnitAudit(): PaperEconomicUnitAudit {
  const computedAt = new Date().toISOString();
  const feeBuf = resolvePaperPolicyFromEnv().feeBuffer;
  const closed = getPaperPortfolio().closedTrades.filter(isClosedTradeWithFiniteRealizedPnl);
  const sorted = [...closed].sort(
    (a, b) => new Date(b.closedAt ?? 0).getTime() - new Date(a.closedAt ?? 0).getTime()
  );
  const samples = sorted.slice(0, SAMPLE_CAP).map((t) => buildSample(t, feeBuf));

  const provKeys = [...PAPER_GRAPH_PROVENANCE_KEYS, "non_graph", "unknown"] as const;
  const byProv = new Map<string, number[]>();
  const exceedCount = new Map<string, number>();
  for (const k of provKeys) {
    byProv.set(k, []);
    exceedCount.set(k, 0);
  }

  let mismatch = 0;
  let missingExit = 0;

  for (const t of closed) {
    const key = provenanceBucket(t);
    const ratios = byProv.get(key) ?? [];
    const fc = t.filledCapital;
    const net = getClosedTradeNetRealizedPnL(t, feeBuf);
    const gross = getClosedTradeGrossRealizedPnL(t);
    if (typeof fc === "number" && fc > 0 && Number.isFinite(net)) {
      ratios.push(net / fc);
      byProv.set(key, ratios);
      if (Math.abs(gross) > fc + 1e-9) {
        exceedCount.set(key, (exceedCount.get(key) ?? 0) + 1);
      }
    }

    const rec = reconstructExitPnL(t);
    if (t.exitPriceEstimate == null || typeof t.exitPriceEstimate !== "number") missingExit += 1;
    if (
      rec.reconstructed != null &&
      Number.isFinite(gross) &&
      Math.abs(rec.reconstructed - gross) >
        Math.max(RECON_EPS_ABS, Math.abs(gross) * RECON_EPS_REL)
    ) {
      mismatch += 1;
    }
  }

  const aggregatesByProvenance: ProvenanceEconomicUnitAgg[] = [];
  for (const k of provKeys) {
    const arr = byProv.get(k) ?? [];
    if (arr.length === 0) {
      aggregatesByProvenance.push({
        provenance: k,
        tradeCount: 0,
        avgRealizedPnlToFilledCapitalRatio: null,
        medianRealizedPnlToFilledCapitalRatio: null,
        maxRealizedPnlToFilledCapitalRatio: null,
        countTradesWhereAbsRealizedPnlExceedsFilledCapital: exceedCount.get(k) ?? 0,
      });
      continue;
    }
    const sum = arr.reduce((a, b) => a + b, 0);
    const med = medianSorted(arr);
    const mx = Math.max(...arr);
    aggregatesByProvenance.push({
      provenance: k,
      tradeCount: arr.length,
      avgRealizedPnlToFilledCapitalRatio: round4(sum / arr.length),
      medianRealizedPnlToFilledCapitalRatio: med != null ? round4(med) : null,
      maxRealizedPnlToFilledCapitalRatio: round4(mx),
      countTradesWhereAbsRealizedPnlExceedsFilledCapital: exceedCount.get(k) ?? 0,
    });
  }

  return {
    computedAt,
    note:
      "simulateExit produz PnL bruto (preço). No fecho paper, o motor persiste gross, taxas estimadas e net; `realizedPnL` no trade = líquido (KPI). Legado: só bruto em `realizedPnL` — net infere-se.",
    realizedPnlFormulaSummary:
      "simulateExit (bruto): pnlPct = (exitPriceEstimate - entryPriceEstimate) / entryPriceEstimate; realizedPnL_gross = filledCapital * pnlPct. No paperTradeEngine: net = gross − (entry+exit fees), fees ≈ filledCapital × feeBuffer por perna.",
    filledCapitalFormulaSummary:
      "simulateEntry: filledCapital = deterministicFill(fillProbability, requestedCapital) = requestedCapital * fillProbability (prob 0–0.95). requestedCapital limitado por portfolioAvailableCapital, capacity.recommendedCapital e 0.1 * opportunity.liquidity.",
    entryExitPriceConventionSummary:
      "entryPriceEstimate = 1 - opportunity.edge na entrada (sem slippage aplicado ao preço no retorno; entrySlippage é registado mas o preço usado no PnL é entryPriceEstimate directo). exitPrice = 1 - latest.edge no fecho quando existe oportunidade actualizada.",
    implicitQuantityAndUsdSummary:
      "Quantidade implícita (sintética): shares ≈ filledCapital / entryPriceEstimate. Valor à saída ≈ shares * exitPriceEstimate; PnL = valor saída - filledCapital, equivalente à fórmula multiplicativa. USD: filledCapital e realizedPnL são a mesma unidade notional USD.",
    feesInSimulatedRealizedPnlSummary:
      "simulateExit continua sem taxas. As taxas estimadas aplicam-se no fecho (`paperTradeEngine`): campos estimated*Fees, netRealizedPnL e realizedPnL líquido alinhados a paperEntryEconomics (feeBuffer × filled por perna × 2).",
    sampleClosedTradeEconomicAudit: samples,
    aggregatesByProvenance,
    totals: {
      closedTradesWithFinitePnlAnalyzed: closed.length,
      countReconstructionMismatch: mismatch,
      countMissingExitPrice: missingExit,
    },
  };
}
