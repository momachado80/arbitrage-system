/**
 * Execution Reality Probe — auditoria de realismo de custos de execução vs proxy de fricção.
 * Não altera digest constraint-first, floors, nem definições de família.
 */

import type { NormalizedMarket } from "./polymarketClient";
import { getAllMarkets } from "./marketDataService";
import {
  buildConstraintFirstEdgeDiscoveryDigest,
  type ConstraintFamilyType,
  type ConstraintFirstFamilyRow,
} from "./constraintFirstEdgeDiscovery";

function envNum(k: string, d: number): number {
  const v = Number(process.env[k]?.trim());
  return Number.isFinite(v) ? v : d;
}

function r6(n: number): number {
  return Math.round(n * 1_000_000) / 1_000_000;
}

function clamp01(x: number): number {
  return Math.max(0, Math.min(1, x));
}

const FEE_COST = 0.0065;
const MAX_REJECTED_FAMILIES = 32;

export type ExecutionRealityVerdictPerFamily =
  | "viable_under_best_case_execution"
  | "viable_under_median_execution"
  | "viable_under_worst_case_execution"
  | "not_viable_under_all_modeled_execution_costs";

export type ExecutionRealityAggregateVerdict =
  | "some_viable_under_best_case_executable_cost_assumptions"
  | "some_viable_under_median_executable_cost_assumptions"
  | "some_viable_under_worst_case_executable_cost_only"
  | "no_families_viable_under_modeled_execution_cost_envelope"
  | "insufficient_sample";

export interface ExecutionRealityFamilyEvaluation {
  familyId: string;
  familyType: ConstraintFamilyType;
  proxyFrictionCost: number;
  observedSpreadCost: number;
  depthAwareCostEstimate: number;
  feeCost: number;
  bestCaseExecutableCost: number;
  medianExecutableCost: number;
  worstCaseExecutableCost: number;
  netUnderBestCaseExecution: number;
  netUnderMedianExecution: number;
  netUnderWorstCaseExecution: number;
  executionRealityVerdictPerFamily: ExecutionRealityVerdictPerFamily;
  supportingNote: string;
}

export interface ExecutionRealityStrongestCandidate {
  familyId: string;
  familyType: ConstraintFamilyType;
  netUnderBestCaseExecution: number;
  netUnderMedianExecution: number;
  proxyFrictionCost: number;
  bestCaseExecutableCost: number;
}

export interface ExecutionRealityProbeDigest {
  probeVersion: "execution-reality-v1";
  readDisclaimer: string;
  executionRealityVerdict: ExecutionRealityAggregateVerdict;
  familiesEvaluated: number;
  familiesViableUnderBestCaseExecution: number;
  familiesViableUnderMedianExecution: number;
  familiesViableUnderWorstCaseExecution: number;
  strongestExecutionRealityCandidate: ExecutionRealityStrongestCandidate | null;
  executionRealitySummaryLine: string;
  families: ExecutionRealityFamilyEvaluation[];
  computedAt: string;
}

function liquidityPenalty(avgLiq: number): number {
  return r6((1 - clamp01(avgLiq / 25_000)) * 0.006);
}

function memberAverages(members: NormalizedMarket[]): { avgLiq: number; avgSp: number } {
  if (members.length === 0) return { avgLiq: 0, avgSp: 0 };
  const avgLiq = members.reduce((a, b) => a + b.liquidity, 0) / members.length;
  const avgSp = members.reduce((a, b) => a + b.spread, 0) / members.length;
  return { avgLiq, avgSp };
}

function evaluateFamily(
  row: ConstraintFirstFamilyRow,
  members: NormalizedMarket[],
  passFloor: number,
): ExecutionRealityFamilyEvaluation | null {
  if (members.length === 0) return null;
  const { avgLiq, avgSp } = memberAverages(members);
  const liqPen = liquidityPenalty(avgLiq);
  const feeCost = FEE_COST;
  const observedSpreadCost = r6(avgSp * 0.5);
  const depthAwareCostEstimate = r6(
    feeCost + avgSp * 0.28 + liqPen + (avgLiq < 8_000 ? 0.004 : 0),
  );
  const bestCaseExecutableCost = r6(feeCost + avgSp * 0.18 + liqPen * 0.55);
  const medianExecutableCost = r6(feeCost + avgSp * 0.32 + liqPen);
  const worstCaseExecutableCost = r6(feeCost + avgSp * 0.58 + liqPen * 1.2);

  const netB = r6(row.rawEdge - bestCaseExecutableCost - row.uncertaintyHaircut - row.modelRiskHaircut);
  const netM = r6(row.rawEdge - medianExecutableCost - row.uncertaintyHaircut - row.modelRiskHaircut);
  const netW = r6(row.rawEdge - worstCaseExecutableCost - row.uncertaintyHaircut - row.modelRiskHaircut);

  let executionRealityVerdictPerFamily: ExecutionRealityVerdictPerFamily =
    "not_viable_under_all_modeled_execution_costs";
  if (netB > passFloor) executionRealityVerdictPerFamily = "viable_under_best_case_execution";
  else if (netM > passFloor) executionRealityVerdictPerFamily = "viable_under_median_execution";
  else if (netW > passFloor) executionRealityVerdictPerFamily = "viable_under_worst_case_execution";

  const supportingNote = `members=${members.length} avgLiq=${r6(avgLiq)} avgSpread=${r6(avgSp)} | proxy_friction=${r6(row.frictionCostEstimate)} vs_best_exec=${bestCaseExecutableCost} | depth_proxy_adds_shallow_liq_penalty_liqPen=${liqPen} | net_scenarios(best/med/worst)=${netB}/${netM}/${netW} vs_floor=${passFloor} (semantic/conc_kill_stack_not_re_simulated)`;

  return {
    familyId: row.familyId,
    familyType: row.familyType,
    proxyFrictionCost: row.frictionCostEstimate,
    observedSpreadCost,
    depthAwareCostEstimate,
    feeCost,
    bestCaseExecutableCost,
    medianExecutableCost,
    worstCaseExecutableCost,
    netUnderBestCaseExecution: netB,
    netUnderMedianExecution: netM,
    netUnderWorstCaseExecution: netW,
    executionRealityVerdictPerFamily,
    supportingNote,
  };
}

export function buildExecutionRealityProbeDigest(): ExecutionRealityProbeDigest {
  const digest = buildConstraintFirstEdgeDiscoveryDigest();
  const passFloor = envNum("CONSTRAINT_FIRST_NET_PASS", 0.0045);
  const allMarkets = getAllMarkets();
  const marketById = new Map<string, NormalizedMarket>();
  for (const m of allMarkets) marketById.set(m.id, m);

  const rejected = digest.families.filter(
    r => !(r.killReason === null && r.netEdgeAfterHaircut > passFloor),
  );
  rejected.sort((a, b) => b.netEdgeAfterHaircut - a.netEdgeAfterHaircut);
  const candidates = rejected.slice(0, MAX_REJECTED_FAMILIES);

  const families: ExecutionRealityFamilyEvaluation[] = [];
  for (const row of candidates) {
    const members: NormalizedMarket[] = [];
    for (const id of row.memberMarketIds) {
      const mm = marketById.get(id);
      if (mm) members.push(mm);
    }
    const ev = evaluateFamily(row, members, passFloor);
    if (ev) families.push(ev);
  }

  const familiesEvaluated = families.length;
  const familiesViableUnderBestCaseExecution = families.filter(
    f => f.executionRealityVerdictPerFamily === "viable_under_best_case_execution",
  ).length;
  const familiesViableUnderMedianExecution = families.filter(
    f => f.executionRealityVerdictPerFamily === "viable_under_median_execution",
  ).length;
  const familiesViableUnderWorstCaseExecution = families.filter(
    f => f.executionRealityVerdictPerFamily === "viable_under_worst_case_execution",
  ).length;

  let strongestExecutionRealityCandidate: ExecutionRealityStrongestCandidate | null = null;
  if (families.length > 0) {
    const top = families.reduce((a, b) => (a.netUnderBestCaseExecution >= b.netUnderBestCaseExecution ? a : b));
    strongestExecutionRealityCandidate = {
      familyId: top.familyId,
      familyType: top.familyType,
      netUnderBestCaseExecution: top.netUnderBestCaseExecution,
      netUnderMedianExecution: top.netUnderMedianExecution,
      proxyFrictionCost: top.proxyFrictionCost,
      bestCaseExecutableCost: top.bestCaseExecutableCost,
    };
  }

  let executionRealityVerdict: ExecutionRealityAggregateVerdict;
  if (familiesEvaluated === 0) {
    executionRealityVerdict = "insufficient_sample";
  } else if (familiesViableUnderBestCaseExecution > 0) {
    executionRealityVerdict = "some_viable_under_best_case_executable_cost_assumptions";
  } else if (familiesViableUnderMedianExecution > 0) {
    executionRealityVerdict = "some_viable_under_median_executable_cost_assumptions";
  } else if (familiesViableUnderWorstCaseExecution > 0) {
    executionRealityVerdict = "some_viable_under_worst_case_executable_cost_only";
  } else {
    executionRealityVerdict = "no_families_viable_under_modeled_execution_cost_envelope";
  }

  const executionRealitySummaryLine = `execution_reality: verdict=${executionRealityVerdict} | evaluated=${familiesEvaluated} viable_best=${familiesViableUnderBestCaseExecution} viable_median=${familiesViableUnderMedianExecution} viable_worst=${familiesViableUnderWorstCaseExecution} | top_best_net=${strongestExecutionRealityCandidate?.netUnderBestCaseExecution ?? "n/a"} (${strongestExecutionRealityCandidate?.familyId ?? "n/a"})`;

  return {
    probeVersion: "execution-reality-v1",
    readDisclaimer:
      "Execution-reality: cenários best/median/worst de custo executável são hipotéticos e não substituem fills reais. Não altera floors nem digest constraint-first. Paper só após evidência de execução.",
    executionRealityVerdict,
    familiesEvaluated,
    familiesViableUnderBestCaseExecution,
    familiesViableUnderMedianExecution,
    familiesViableUnderWorstCaseExecution,
    strongestExecutionRealityCandidate,
    executionRealitySummaryLine,
    families,
    computedAt: new Date().toISOString(),
  };
}
