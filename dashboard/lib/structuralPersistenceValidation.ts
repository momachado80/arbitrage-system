/**
 * Structural Persistence Validation — validação prospectiva em camadas de granularidade.
 * Testa persistência primeiro em níveis coarse, depois intermediate, depois fine.
 * Não cria challenger; não altera execução ou thresholds.
 */

import type { ClosedTradeAuditEntry } from "./shadowClosedTradeAudit";
import { SAMPLE_GATE_CONFIG } from "./structuralResearchV2";
import type { SubsetMetrics } from "./structuralResearchV2";

export type GranularityLevel = "coarse" | "intermediate" | "fine";

/** Gates por granularidade: coarse mais permissivo, fine mais exigente */
export const PERSISTENCE_GATE_BY_LEVEL = {
  coarse: { minDiscoverySample: 5, minValidationSample: 5 },
  intermediate: { minDiscoverySample: 5, minValidationSample: 5 },
  fine: { minDiscoverySample: 8, minValidationSample: 8 },
} as const;

/** Thresholds compartilhados */
export const PERSISTENCE_THRESHOLDS = {
  materialDegradationThresholdPerTrade: 0.01,
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

export interface PersistenceSubsetResult {
  subsetKey: string;
  granularity: GranularityLevel;
  discoveryMetrics: SubsetMetrics;
  validationMetrics: SubsetMetrics;
  deltaAvgPnL: number;
  deltaWinRate: number;
  maintainedLessDestructive: boolean;
  degradedMaterially: boolean;
  collapsed: boolean;
  persistentLeastBad: boolean;
  discoverySampleSufficient: boolean;
  validationSampleSufficient: boolean;
  eligibleForNarrowChallenger: boolean;
}

export interface PersistenceLevelBlock {
  granularity: GranularityLevel;
  definition: string;
  gateConfig: { minDiscoverySample: number; minValidationSample: number };
  eligibleSubsets: PersistenceSubsetResult[];
  rejectedForLowSample: string[];
  persistentLeastBad: string[];
  degradedSubsets: string[];
  collapsedSubsets: string[];
  eligibleForNarrowChallenger: string[];
}

export interface StructuralPersistenceValidation {
  methodologySummary: string;
  discoveryWindowDefinition: string;
  validationWindowDefinition: string;
  newTradesCount: number;
  discoveryCount: number;
  validationCount: number;
  coarsePersistence: PersistenceLevelBlock;
  intermediatePersistence: PersistenceLevelBlock;
  finePersistence: PersistenceLevelBlock;
  persistenceScoreExplanation: string;
  dataScope: "new_only";
}

function assignCoarse(
  entries: ClosedTradeAuditEntry[]
): { byPair: Map<string, ClosedTradeAuditEntry[]>; byEdge: Map<string, ClosedTradeAuditEntry[]>; byFill: Map<string, ClosedTradeAuditEntry[]> } {
  const byPair = new Map<string, ClosedTradeAuditEntry[]>();
  const byEdge = new Map<string, ClosedTradeAuditEntry[]>();
  const byFill = new Map<string, ClosedTradeAuditEntry[]>();

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
    push(byEdge, eb, e);
    push(byFill, fb, e);
  }
  return { byPair, byEdge, byFill };
}

function assignIntermediate(
  entries: ClosedTradeAuditEntry[]
): { byPairEdge: Map<string, ClosedTradeAuditEntry[]>; byPairFill: Map<string, ClosedTradeAuditEntry[]> } {
  const byPairEdge = new Map<string, ClosedTradeAuditEntry[]>();
  const byPairFill = new Map<string, ClosedTradeAuditEntry[]>();

  const push = (map: Map<string, ClosedTradeAuditEntry[]>, key: string, e: ClosedTradeAuditEntry) => {
    const arr = map.get(key) ?? [];
    arr.push(e);
    map.set(key, arr);
  };

  for (const e of entries) {
    const pk = e.pairKey ?? "_no_pair";
    const eb = edgeBucket(e.capturableEdgeAtEntry ?? 0);
    const fb = fillBucket(e.fillRatio);
    push(byPairEdge, `${pk}|${eb}`, e);
    push(byPairFill, `${pk}|${fb}`, e);
  }
  return { byPairEdge, byPairFill };
}

function assignFine(entries: ClosedTradeAuditEntry[]): Map<string, ClosedTradeAuditEntry[]> {
  const byPairEdgeFill = new Map<string, ClosedTradeAuditEntry[]>();
  for (const e of entries) {
    const pk = e.pairKey ?? "_no_pair";
    const eb = edgeBucket(e.capturableEdgeAtEntry ?? 0);
    const fb = fillBucket(e.fillRatio);
    const key = `${pk}|${eb}|${fb}`;
    const arr = byPairEdgeFill.get(key) ?? [];
    arr.push(e);
    byPairEdgeFill.set(key, arr);
  }
  return byPairEdgeFill;
}

function runLevelPersistence(
  discMap: Map<string, ClosedTradeAuditEntry[]>,
  valMap: Map<string, ClosedTradeAuditEntry[]>,
  granularity: GranularityLevel,
  definition: string,
  globalDiscoveryAvg: number
): PersistenceLevelBlock {
  const gate = PERSISTENCE_GATE_BY_LEVEL[granularity];
  const thresh = PERSISTENCE_THRESHOLDS;

  const eligible: PersistenceSubsetResult[] = [];
  const rejectedLow: string[] = [];
  const persistentLeastBadList: string[] = [];
  const degradedList: string[] = [];
  const collapsedList: string[] = [];
  const eligibleNarrowList: string[] = [];

  const allKeys = new Set<string>();
  for (const k of Array.from(discMap.keys())) allKeys.add(k);
  for (const k of Array.from(valMap.keys())) allKeys.add(k);

  for (const key of Array.from(allKeys)) {
    const fullKey = `${granularity}:${key}`;
    const discEntries = discMap.get(key) ?? [];
    const valEntries = valMap.get(key) ?? [];
    const discM = computeMetrics(discEntries);
    const valM = computeMetrics(valEntries);

    const discoverySampleSufficient = discM.tradeCount >= gate.minDiscoverySample;
    const validationSampleSufficient = valM.tradeCount >= gate.minValidationSample;
    const rejectedForLowSample = !discoverySampleSufficient || !validationSampleSufficient;

    if (rejectedForLowSample) {
      rejectedLow.push(fullKey);
      continue;
    }

    const materialDegradation =
      valM.avgRealizedPnL < discM.avgRealizedPnL - thresh.materialDegradationThresholdPerTrade;
    const collapsedCheck =
      discM.avgRealizedPnL > thresh.collapseDiscoveryCeiling &&
      valM.avgRealizedPnL < thresh.collapseValidationFloor;
    const maintainedLessDestructive = !materialDegradation && !collapsedCheck;
    const persistentLeastBad = maintainedLessDestructive;
    const qualityDifferentiated = discM.avgRealizedPnL > globalDiscoveryAvg;
    const eligibleForNarrowChallenger =
      persistentLeastBad &&
      discM.tradeCount >= SAMPLE_GATE_CONFIG.minForExploratoryRanking &&
      qualityDifferentiated;

    const result: PersistenceSubsetResult = {
      subsetKey: key,
      granularity,
      discoveryMetrics: discM,
      validationMetrics: valM,
      deltaAvgPnL: valM.avgRealizedPnL - discM.avgRealizedPnL,
      deltaWinRate: valM.winRate - discM.winRate,
      maintainedLessDestructive,
      degradedMaterially: materialDegradation,
      collapsed: collapsedCheck,
      persistentLeastBad,
      discoverySampleSufficient,
      validationSampleSufficient,
      eligibleForNarrowChallenger,
    };

    eligible.push(result);
    if (persistentLeastBad) persistentLeastBadList.push(fullKey);
    if (materialDegradation) degradedList.push(fullKey);
    if (collapsedCheck) collapsedList.push(fullKey);
    if (eligibleForNarrowChallenger) eligibleNarrowList.push(fullKey);
  }

  return {
    granularity,
    definition,
    gateConfig: gate,
    eligibleSubsets: eligible,
    rejectedForLowSample: rejectedLow,
    persistentLeastBad: persistentLeastBadList,
    degradedSubsets: degradedList,
    collapsedSubsets: collapsedList,
    eligibleForNarrowChallenger: eligibleNarrowList,
  };
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

  const globalDiscM = computeMetrics(discoveryEntries);
  const globalDiscoveryAvg = globalDiscM.avgRealizedPnL;

  const coarseDisc = assignCoarse(discoveryEntries);
  const coarseVal = assignCoarse(validationEntries);
  const intermediateDisc = assignIntermediate(discoveryEntries);
  const intermediateVal = assignIntermediate(validationEntries);
  const fineDisc = assignFine(discoveryEntries);
  const fineVal = assignFine(validationEntries);

  const coarsePersistence = ((): PersistenceLevelBlock => {
    const pairBlock = runLevelPersistence(
      coarseDisc.byPair,
      coarseVal.byPair,
      "coarse",
      "pair (by pairKey)",
      globalDiscoveryAvg
    );
    const edgeBlock = runLevelPersistence(
      coarseDisc.byEdge,
      coarseVal.byEdge,
      "coarse",
      "edgeBucket (by capturable edge bucket)",
      globalDiscoveryAvg
    );
    const fillBlock = runLevelPersistence(
      coarseDisc.byFill,
      coarseVal.byFill,
      "coarse",
      "fillBucket (by fill ratio bucket)",
      globalDiscoveryAvg
    );
    return {
      granularity: "coarse",
      definition: "pair, edgeBucket, fillBucket (single dimension)",
      gateConfig: PERSISTENCE_GATE_BY_LEVEL.coarse,
      eligibleSubsets: [
        ...pairBlock.eligibleSubsets,
        ...edgeBlock.eligibleSubsets,
        ...fillBlock.eligibleSubsets,
      ],
      rejectedForLowSample: [
        ...pairBlock.rejectedForLowSample,
        ...edgeBlock.rejectedForLowSample,
        ...fillBlock.rejectedForLowSample,
      ],
      persistentLeastBad: [
        ...pairBlock.persistentLeastBad,
        ...edgeBlock.persistentLeastBad,
        ...fillBlock.persistentLeastBad,
      ],
      degradedSubsets: [...pairBlock.degradedSubsets, ...edgeBlock.degradedSubsets, ...fillBlock.degradedSubsets],
      collapsedSubsets: [...pairBlock.collapsedSubsets, ...edgeBlock.collapsedSubsets, ...fillBlock.collapsedSubsets],
      eligibleForNarrowChallenger: [
        ...pairBlock.eligibleForNarrowChallenger,
        ...edgeBlock.eligibleForNarrowChallenger,
        ...fillBlock.eligibleForNarrowChallenger,
      ],
    };
  })();

  const intermediatePersistence = ((): PersistenceLevelBlock => {
    const pairEdgeBlock = runLevelPersistence(
      intermediateDisc.byPairEdge,
      intermediateVal.byPairEdge,
      "intermediate",
      "pair×edge",
      globalDiscoveryAvg
    );
    const pairFillBlock = runLevelPersistence(
      intermediateDisc.byPairFill,
      intermediateVal.byPairFill,
      "intermediate",
      "pair×fill",
      globalDiscoveryAvg
    );
    return {
      granularity: "intermediate",
      definition: "pair×edge, pair×fill",
      gateConfig: PERSISTENCE_GATE_BY_LEVEL.intermediate,
      eligibleSubsets: [...pairEdgeBlock.eligibleSubsets, ...pairFillBlock.eligibleSubsets],
      rejectedForLowSample: [
        ...pairEdgeBlock.rejectedForLowSample,
        ...pairFillBlock.rejectedForLowSample,
      ],
      persistentLeastBad: [
        ...pairEdgeBlock.persistentLeastBad,
        ...pairFillBlock.persistentLeastBad,
      ],
      degradedSubsets: [...pairEdgeBlock.degradedSubsets, ...pairFillBlock.degradedSubsets],
      collapsedSubsets: [...pairEdgeBlock.collapsedSubsets, ...pairFillBlock.collapsedSubsets],
      eligibleForNarrowChallenger: [
        ...pairEdgeBlock.eligibleForNarrowChallenger,
        ...pairFillBlock.eligibleForNarrowChallenger,
      ],
    };
  })();

  const finePersistence = runLevelPersistence(
    fineDisc,
    fineVal,
    "fine",
    "pair×edge×fill",
    globalDiscoveryAvg
  );

  return {
    methodologySummary:
      "Persistence validation in granularity layers. Coarse = pair, edgeBucket, fillBucket. Intermediate = pair×edge, pair×fill. Fine = pair×edge×fill. " +
      "Each level has its own sample gates (coarse/intermediate 5/5, fine 8/8). " +
      "persistentLeastBad = maintained less destructive behavior across windows. " +
      "eligibleForNarrowChallenger = persistent + discovery avgPnL > global + n>=10.",
    discoveryWindowDefinition: "First 50% of new trades by closedAt (chronological order)",
    validationWindowDefinition: "Last 50% of new trades by closedAt (chronological order)",
    newTradesCount: newEntries.length,
    discoveryCount: discoveryEntries.length,
    validationCount: validationEntries.length,
    coarsePersistence,
    intermediatePersistence,
    finePersistence,
    persistenceScoreExplanation:
      "persistentLeastBad: validation not materially worse than discovery. degraded: validation avgPnL dropped > 0.01/trade. " +
      "collapsed: discovery avgPnL > -0.02, validation < -0.05. eligibleForNarrowChallenger: persistent + discovery > global avg + n>=10.",
    dataScope: "new_only",
  };
}
