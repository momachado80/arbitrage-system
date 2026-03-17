/**
 * Structural Persistence Validation — validação prospectiva obrigatória para subsets estruturais.
 * Nenhum subset é elegível a narrow challenger sem prova de persistência fora da janela de descoberta.
 * Não cria challenger; não altera execução ou thresholds.
 */

import type { ClosedTradeAuditEntry } from "./shadowClosedTradeAudit";
import { SAMPLE_GATE_CONFIG } from "./structuralResearchV2";
import type { SubsetMetrics } from "./structuralResearchV2";

/** Gates para persistência: amostra mínima em cada janela */
export const PERSISTENCE_GATE_CONFIG = {
  minDiscoverySample: 5,
  minValidationSample: 5,
  /** Degradação material: validation avgPnL < discovery avgPnL - este valor por trade */
  materialDegradationThresholdPerTrade: 0.01,
  /** Colapso: discovery era menos destrutivo (avgPnL > -0.02) e validation avgPnL < -0.05 */
  collapseValidationFloor: -0.05,
  collapseDiscoveryCeiling: -0.02,
} as const;

function edgeBucket(edge: number): string {
  if (edge < 0.005) return "0-0.5%";
  if (edge < 0.01) return "0.5-1%";
  if (edge < 0.02) return "1-2%";
  if (edge < 0.05) return "2-5%";
  return ">5%";
}

function fillBucket(r: number | null | undefined): string {
  if (r == null || r < 0) return "_unknown";
  if (r <= 0.1) return "0-0.1";
  if (r <= 0.25) return "0.1-0.25";
  if (r <= 0.5) return "0.25-0.5";
  if (r <= 0.75) return "0.5-0.75";
  return "0.75-1.0";
}

function computeMetrics(entries: ClosedTradeAuditEntry[]): SubsetMetrics {
  const n = entries.length;
  if (n === 0) {
    return {
      tradeCount: 0,
      avgRealizedPnL: 0,
      medianRealizedPnL: 0,
      winRate: 0,
      avgFillRatio: 0,
      avgCapturableEdgeAtEntry: 0,
      avgObservedEdgeAtEntry: 0,
      avgHoldingTimeMs: 0,
    };
  }
  const pnls = entries.map((e) => e.realizedPnL);
  const sorted = [...pnls].sort((a, b) => a - b);
  const m = Math.floor(n / 2);
  const medianRealizedPnL = n % 2 ? sorted[m] : (sorted[m - 1] + sorted[m]) / 2;
  const wins = entries.filter((e) => e.realizedPnL > 0).length;
  const fillRatios = entries
    .map((e) => e.fillRatio)
    .filter((v): v is number => v != null && typeof v === "number");

  return {
    tradeCount: n,
    avgRealizedPnL: pnls.reduce((s, x) => s + x, 0) / n,
    medianRealizedPnL,
    winRate: wins / n,
    avgFillRatio: fillRatios.length ? fillRatios.reduce((a, b) => a + b, 0) / fillRatios.length : 0,
    avgCapturableEdgeAtEntry: entries.reduce((a, e) => a + (e.capturableEdgeAtEntry ?? 0), 0) / n,
    avgObservedEdgeAtEntry: entries.reduce((a, e) => a + (e.observedEdgeAtEntry ?? 0), 0) / n,
    avgHoldingTimeMs: entries.reduce((a, e) => a + (e.holdingTimeMs ?? 0), 0) / n,
  };
}

export type SubsetLevel = "pair" | "pair_x_edge" | "pair_x_fill" | "pair_x_edge_x_fill";

export interface PersistenceSubsetResult {
  subsetKey: string;
  level: SubsetLevel;
  discoveryMetrics: SubsetMetrics;
  validationMetrics: SubsetMetrics;
  deltaAvgPnL: number;
  deltaWinRate: number;
  maintainedLessDestructive: boolean;
  degradedMaterially: boolean;
  collapsed: boolean;
  surviving: boolean;
  rejectedForLowSample: boolean;
  rejectedForNoPersistence: boolean;
  discoverySampleSufficient: boolean;
  validationSampleSufficient: boolean;
  eligibleForNarrowChallenger: boolean;
}

export interface StructuralPersistenceValidation {
  methodologySummary: string;
  discoveryWindowDefinition: string;
  validationWindowDefinition: string;
  sampleGateConfig: typeof PERSISTENCE_GATE_CONFIG;
  newTradesCount: number;
  discoveryCount: number;
  validationCount: number;
  eligibleSubsets: PersistenceSubsetResult[];
  rejectedForLowSample: string[];
  rejectedForNoPersistence: string[];
  degradedSubsets: string[];
  survivingSubsets: string[];
  collapsedSubsets: string[];
  persistenceScoreExplanation: string;
  dataScope: "new_only";
}

function assignToSubsets(
  entries: ClosedTradeAuditEntry[]
): Map<string, ClosedTradeAuditEntry[]>[] {
  const byPair = new Map<string, ClosedTradeAuditEntry[]>();
  const byPairEdge = new Map<string, ClosedTradeAuditEntry[]>();
  const byPairFill = new Map<string, ClosedTradeAuditEntry[]>();
  const byPairEdgeFill = new Map<string, ClosedTradeAuditEntry[]>();

  const push = (map: Map<string, ClosedTradeAuditEntry[]>, key: string, e: ClosedTradeAuditEntry) => {
    const arr = map.get(key) ?? [];
    arr.push(e);
    map.set(key, arr);
  };

  for (const e of entries) {
    const pk = e.pairKey ?? "_no_pair";
    const eb = edgeBucket(e.capturableEdgeAtEntry ?? 0);
    const fb = fillBucket(e.fillRatio);
    push(byPair, pk, e);
    push(byPairEdge, `${pk}|${eb}`, e);
    push(byPairFill, `${pk}|${fb}`, e);
    push(byPairEdgeFill, `${pk}|${eb}|${fb}`, e);
  }
  return [byPair, byPairEdge, byPairFill, byPairEdgeFill];
}

export function runStructuralPersistenceValidation(
  entries: ClosedTradeAuditEntry[],
  rehydratedIds: ReadonlySet<string>
): StructuralPersistenceValidation {
  const newEntries = entries.filter((e) => !rehydratedIds.has(e.tradeId));
  const sorted = [...newEntries].sort(
    (a, b) => new Date(a.closedAt).getTime() - new Date(b.closedAt).getTime()
  );
  const mid = Math.floor(sorted.length / 2);
  const discoveryEntries = sorted.slice(0, mid);
  const validationEntries = sorted.slice(mid);

  const gate = PERSISTENCE_GATE_CONFIG;
  const rejectedLow: string[] = [];
  const rejectedNoPersist: string[] = [];
  const degraded: string[] = [];
  const surviving: string[] = [];
  const collapsed: string[] = [];
  const eligible: PersistenceSubsetResult[] = [];

  const levels: SubsetLevel[] = ["pair", "pair_x_edge", "pair_x_fill", "pair_x_edge_x_fill"];
  const [discByPair, discByPairEdge, discByPairFill, discByPairEdgeFill] = assignToSubsets(discoveryEntries);
  const [valByPair, valByPairEdge, valByPairFill, valByPairEdgeFill] = assignToSubsets(validationEntries);

  const maps = [
    { disc: discByPair, val: valByPair, level: "pair" as SubsetLevel },
    { disc: discByPairEdge, val: valByPairEdge, level: "pair_x_edge" as SubsetLevel },
    { disc: discByPairFill, val: valByPairFill, level: "pair_x_fill" as SubsetLevel },
    { disc: discByPairEdgeFill, val: valByPairEdgeFill, level: "pair_x_edge_x_fill" as SubsetLevel },
  ];

  const seen = new Set<string>();

  for (const { disc, val, level } of maps) {
    const allKeys = new Set<string>();
    for (const k of Array.from(disc.keys())) allKeys.add(k);
    for (const k of Array.from(val.keys())) allKeys.add(k);
    for (const key of Array.from(allKeys)) {
      const fullKey = `${level}:${key}`;
      if (seen.has(fullKey)) continue;
      seen.add(fullKey);

      const discEntries = disc.get(key) ?? [];
      const valEntries = val.get(key) ?? [];
      const discM = computeMetrics(discEntries);
      const valM = computeMetrics(valEntries);

      const discoverySampleSufficient = discM.tradeCount >= gate.minDiscoverySample;
      const validationSampleSufficient = valM.tradeCount >= gate.minValidationSample;
      const rejectedForLowSample = !discoverySampleSufficient || !validationSampleSufficient;
      if (rejectedForLowSample) {
        if (!discoverySampleSufficient) rejectedLow.push(fullKey);
        else if (!validationSampleSufficient) rejectedLow.push(fullKey);
        continue;
      }

      const deltaAvgPnL = valM.avgRealizedPnL - discM.avgRealizedPnL;
      const deltaWinRate = valM.winRate - discM.winRate;
      const materialDegradation =
        valM.avgRealizedPnL < discM.avgRealizedPnL - gate.materialDegradationThresholdPerTrade;
      const collapsedCheck =
        discM.avgRealizedPnL > gate.collapseDiscoveryCeiling &&
        valM.avgRealizedPnL < gate.collapseValidationFloor;
      const maintainedLessDestructive = !materialDegradation && !collapsedCheck;
      const degradedMaterially = materialDegradation;
      const collapsed_flag = collapsedCheck;
      const surviving_flag = maintainedLessDestructive && discM.avgRealizedPnL > -0.1;
      const rejectedForNoPersistence = !maintainedLessDestructive;
      const eligibleForNarrowChallenger =
        discoverySampleSufficient &&
        validationSampleSufficient &&
        maintainedLessDestructive &&
        discM.tradeCount >= SAMPLE_GATE_CONFIG.minForExploratoryRanking;

      if (rejectedForNoPersistence) rejectedNoPersist.push(fullKey);
      if (degradedMaterially) degraded.push(fullKey);
      if (collapsed_flag) collapsed.push(fullKey);
      if (surviving_flag) surviving.push(fullKey);

      eligible.push({
        subsetKey: key,
        level,
        discoveryMetrics: discM,
        validationMetrics: valM,
        deltaAvgPnL,
        deltaWinRate,
        maintainedLessDestructive,
        degradedMaterially,
        collapsed: collapsed_flag,
        surviving: surviving_flag,
        rejectedForLowSample: false,
        rejectedForNoPersistence,
        discoverySampleSufficient,
        validationSampleSufficient,
        eligibleForNarrowChallenger,
      });
    }
  }

  return {
    methodologySummary:
      "Prospective validation: new trades split 50/50 by closedAt. First half = discovery, second half = validation. " +
      "Subset must have minDiscoverySample and minValidationSample in each window. " +
      "Maintained = validation not materially worse than discovery. Degraded = validation avgPnL dropped > threshold. " +
      "Collapsed = discovery was less destructive, validation reversed. Surviving = maintained and discovery avgPnL > -0.1. " +
      "Eligible for narrow challenger = discovery+validation sufficient, maintained, discovery rank ok.",
    discoveryWindowDefinition: "First 50% of new trades by closedAt (chronological order)",
    validationWindowDefinition: "Last 50% of new trades by closedAt (chronological order)",
    sampleGateConfig: gate,
    newTradesCount: newEntries.length,
    discoveryCount: discoveryEntries.length,
    validationCount: validationEntries.length,
    eligibleSubsets: eligible,
    rejectedForLowSample: Array.from(new Set(rejectedLow)),
    rejectedForNoPersistence: rejectedNoPersist,
    degradedSubsets: degraded,
    survivingSubsets: surviving,
    collapsedSubsets: collapsed,
    persistenceScoreExplanation:
      "maintainedLessDestructive: validation avgPnL not materially worse. degradedMaterially: validation dropped > 0.01/trade vs discovery. " +
      "collapsed: discovery avgPnL > -0.02, validation < -0.05. surviving: maintained and discovery avgPnL > -0.1. " +
      "eligibleForNarrowChallenger: sufficient sample in both windows, maintained, discovery n>=10.",
    dataScope: "new_only",
  };
}
