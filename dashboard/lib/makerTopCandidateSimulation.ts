/**
 * Top maker candidates — simulação offline mais estrita (fila, fills passivos,
 * adverse selection, inventário, unwind). Sem live quoting nem ordens reais.
 * Apenas os top 3 por estimatedMakerNet do makerInventoryProbe.
 */

import { getAllMarkets } from "./marketDataService";
import { buildMakerInventoryProbeDigest, type MakerCandidateMarket } from "./makerInventoryProbe";
import type { NormalizedMarket } from "./polymarketClient";

function r6(n: number): number {
  return Math.round(n * 1_000_000) / 1_000_000;
}

function clamp01(x: number): number {
  return Math.max(0, Math.min(1, x));
}

export type MakerTopSimVerdict =
  | "no_candidate_survives_simulation"
  | "weak_candidate_survives_only_in_best_case"
  | "one_candidate_viable_in_median_case"
  | "multiple_candidates_viable_in_median_case";

export type PerCandidateSimulationVerdict = "median_viable" | "best_case_only" | "fails_simulation";

export interface MakerTopSimCandidateResult {
  marketId: string;
  marketTitle: string;
  sourceEstimatedMakerNet: number;
  simulatedBestCasePnL: number;
  simulatedMedianCasePnL: number;
  simulatedAdverseCasePnL: number;
  simulatedFillRate: number;
  simulatedAdverseSelectionCost: number;
  simulatedInventoryDrag: number;
  simulatedNetPerTurn: number;
  simulatedNetAcrossTurns: number;
  simulationVerdict: PerCandidateSimulationVerdict;
  supportingNote: string;
}

export interface StrongestSimulatedCandidate {
  marketId: string;
  marketTitle: string;
  simulatedMedianCasePnL: number;
}

export interface MakerTopSimDigest {
  probeVersion: "maker-top-sim-v1";
  readDisclaimer: string;
  makerTopSimVerdict: MakerTopSimVerdict;
  candidatesSimulated: number;
  viableCandidatesCount: number;
  strongestSimulatedCandidate: StrongestSimulatedCandidate | null;
  makerTopSimSummaryLine: string;
  candidateSimResults: MakerTopSimCandidateResult[];
  computedAt: string;
}

const TURNS = 22;
const MEDIAN_VIABLE_CUMULATIVE = 0.014;

function queueCompetition(m: NormalizedMarket): number {
  if (m.liquidity <= 0) return 1;
  return clamp01(m.volume / (m.liquidity * 6 + 500));
}

type Branch = "best" | "median" | "adverse";

const BRANCH: Record<Branch, { fill: number; adverse: number; inventory: number }> = {
  best: { fill: 1.16, adverse: 0.7, inventory: 0.76 },
  median: { fill: 1, adverse: 1, inventory: 1 },
  adverse: { fill: 0.56, adverse: 1.52, inventory: 1.38 },
};

function runBranch(
  row: MakerCandidateMarket,
  m: NormalizedMarket | undefined,
  branch: Branch,
): {
  cumulativeNet: number;
  avgFill: number;
  avgAdverse: number;
  avgInv: number;
  avgNetPerTurn: number;
} {
  const qc = m ? queueCompetition(m) : 0.45;
  const spread = row.observedSpread;
  const b = BRANCH[branch];

  const fillBase = clamp01(row.fillPlausibilityMaker * (1 - 0.42 * qc) * 0.64 * b.fill);

  let invAccum = 0;
  let cumulative = 0;
  let sumFill = 0;
  let sumAdv = 0;
  let sumInv = 0;

  for (let t = 1; t <= TURNS; t++) {
    const queueDrag = 1 - 0.05 * Math.min(3.5, invAccum);
    const fill = clamp01(fillBase * queueDrag * (1 - 0.018 * (t - 1)));
    const gross = spread * 0.11 * fill;

    const adverseHitProb = clamp01((0.2 + 0.5 * qc) * b.adverse);
    const adverseCost = adverseHitProb * (spread * 0.31 + 0.007);

    invAccum = invAccum * 0.86 + fill * 0.24;
    const invDrag =
      (row.inventoryRiskProxy * b.inventory * (1 + 0.055 * invAccum) +
        spread * 0.028 * invAccum * b.inventory) *
      0.92;

    const unwind = 0.0014 + 0.00055 * invAccum;
    const hedge = row.estimatedHedgeCost;
    const fee = 0.003;

    const net = gross - adverseCost - invDrag - unwind - hedge - fee;
    cumulative += net;
    sumFill += fill;
    sumAdv += adverseCost;
    sumInv += invDrag;
  }

  return {
    cumulativeNet: r6(cumulative),
    avgFill: r6(sumFill / TURNS),
    avgAdverse: r6(sumAdv / TURNS),
    avgInv: r6(sumInv / TURNS),
    avgNetPerTurn: r6(cumulative / TURNS),
  };
}

function perCandidateVerdict(medianCum: number, bestCum: number): PerCandidateSimulationVerdict {
  if (medianCum >= MEDIAN_VIABLE_CUMULATIVE) return "median_viable";
  if (bestCum > 0) return "best_case_only";
  return "fails_simulation";
}

export function buildMakerTopCandidateSimulationDigest(): MakerTopSimDigest {
  const inventoryDigest = buildMakerInventoryProbeDigest();
  const top = inventoryDigest.candidateMarkets.slice(0, 3);
  const byId = new Map(getAllMarkets().map(m => [m.id, m]));

  const candidateSimResults: MakerTopSimCandidateResult[] = top.map(row => {
    const m = byId.get(row.marketId);
    const best = runBranch(row, m, "best");
    const median = runBranch(row, m, "median");
    const adverse = runBranch(row, m, "adverse");

    const simulationVerdict = perCandidateVerdict(median.cumulativeNet, best.cumulativeNet);

    const supportingNote = `offline_sim:${TURNS}_turns | stricter_capture~spread*0.11*fill | queue_penalty~vol/(liq*6+500) | no_live_orders | fragility_best_median_adverse`;

    return {
      marketId: row.marketId,
      marketTitle: row.marketTitle,
      sourceEstimatedMakerNet: row.estimatedMakerNet,
      simulatedBestCasePnL: best.cumulativeNet,
      simulatedMedianCasePnL: median.cumulativeNet,
      simulatedAdverseCasePnL: adverse.cumulativeNet,
      simulatedFillRate: median.avgFill,
      simulatedAdverseSelectionCost: median.avgAdverse,
      simulatedInventoryDrag: median.avgInv,
      simulatedNetPerTurn: median.avgNetPerTurn,
      simulatedNetAcrossTurns: median.cumulativeNet,
      simulationVerdict,
      supportingNote,
    };
  });

  const viableCandidatesCount = candidateSimResults.filter(r => r.simulationVerdict === "median_viable").length;

  let strongestSimulatedCandidate: StrongestSimulatedCandidate | null = null;
  if (candidateSimResults.length > 0) {
    const sorted = [...candidateSimResults].sort((a, b) => b.simulatedMedianCasePnL - a.simulatedMedianCasePnL);
    const topM = sorted[0];
    strongestSimulatedCandidate = {
      marketId: topM.marketId,
      marketTitle: topM.marketTitle,
      simulatedMedianCasePnL: topM.simulatedMedianCasePnL,
    };
  }

  let makerTopSimVerdict: MakerTopSimVerdict;
  if (candidateSimResults.length === 0) {
    makerTopSimVerdict = "no_candidate_survives_simulation";
  } else if (viableCandidatesCount >= 2) {
    makerTopSimVerdict = "multiple_candidates_viable_in_median_case";
  } else if (viableCandidatesCount === 1) {
    makerTopSimVerdict = "one_candidate_viable_in_median_case";
  } else if (candidateSimResults.some(r => r.simulationVerdict === "best_case_only")) {
    makerTopSimVerdict = "weak_candidate_survives_only_in_best_case";
  } else {
    makerTopSimVerdict = "no_candidate_survives_simulation";
  }

  const makerTopSimSummaryLine = `maker_top_sim: verdict=${makerTopSimVerdict} | simulated=${candidateSimResults.length} median_viable=${viableCandidatesCount} | strongest_median_pnl=${strongestSimulatedCandidate?.simulatedMedianCasePnL ?? "n/a"}`;

  return {
    probeVersion: "maker-top-sim-v1",
    readDisclaimer:
      "Simulação offline nos top 3 por estimatedMakerNet do maker-inventory probe. Custos e fills são proxies; não substitui tape nem coloca ordens. Objectivo: falsificar candidatos fortes sob stress de fila e inventário.",
    makerTopSimVerdict,
    candidatesSimulated: candidateSimResults.length,
    viableCandidatesCount,
    strongestSimulatedCandidate,
    makerTopSimSummaryLine,
    candidateSimResults,
    computedAt: new Date().toISOString(),
  };
}
