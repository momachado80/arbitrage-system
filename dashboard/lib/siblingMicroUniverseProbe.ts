/**
 * Universo observacional irmão: subconjunto de microbuckets recorrentes/estáveis da família
 * other:price_above:>3m, derivado apenas de digests já existentes (catalog / economics / execution)
 * e do estado paper já persistido. Não altera scans, gates nem ficheiros de estado existentes.
 */

import { buildCatalogPocketDigest } from "./catalogPocketProbe";
import type { FamilyPocketRow } from "./catalogPocketProbe";
import { buildPocketEconomicsDigest } from "./pocketEconomicsProbe";
import type { PocketEconomicsDigest } from "./pocketEconomicsProbe";
import { buildPocketExecutionDigest } from "./pocketExecutionProbe";
import type { PocketExecutionDigest } from "./pocketExecutionProbe";
import {
  getMinimalPaperExecutionEntriesReadonly,
  type MinimalPaperEntry,
} from "./minimalPaperExecutionProbe";
import {
  buildMicroEdgeAssessmentFromEntries,
  type MicroEdgeAssessmentDigest,
} from "./minimalPaperMicroEdgeAssessment";

const TARGET_FAMILY_KEY = "other:price_above:>3m" as const;

function envNum(k: string, d: number): number {
  const v = Number(process.env[k]?.trim());
  return Number.isFinite(v) ? v : d;
}

function r4(n: number): number {
  return Math.round(n * 10000) / 10000;
}

function parseWhitelist(): Set<string> | null {
  const raw = process.env.SIBLING_MICRO_BUCKET_WHITELIST?.trim();
  if (!raw) return null;
  const keys = raw.split(/[,|]/g).map(s => s.trim()).filter(s => s.length > 0);
  return keys.length > 0 ? new Set(keys) : null;
}

export interface SiblingMicroBucketSampleRow {
  microBucketKey: string;
  inCatalogPocketStabilityKey: boolean;
  inEconomicsStableMicroBuckets: boolean;
  inEconomicsRepeatedMicroBuckets: boolean;
  eligibleCountEconomics: number | null;
  inExecutionStableMicroKeysAtScan: boolean;
  eligibleMarketCountExecution: number | null;
  betweenSnapshotsStableExecution: boolean | null;
}

export interface SiblingMicroUniverseComparisonDigest {
  mainUniverseMicroEdgeSummary: {
    microEdgeReadVerdict: string;
    eligibleClosedEpisodesForMicroRead: number;
    microPositiveRate: number | null;
    microNegativeRate: number | null;
    microNeutralEpisodes: number;
    repeatedNeutralPattern: boolean;
  };
  siblingUniverseMicroEdgeSummary: {
    microEdgeReadVerdict: string;
    eligibleClosedEpisodesForMicroRead: number;
    microPositiveRate: number | null;
    microNegativeRate: number | null;
    microNeutralEpisodes: number;
    repeatedNeutralPattern: boolean;
  };
  deltas: {
    microPositiveRateDelta: number | null;
    microNegativeRateDelta: number | null;
    eligibleClosedForMicroReadDelta: number;
  };
  interpretationNotes: string[];
}

export interface SiblingMicroUniverseDigest {
  computedAt: string;
  probeVersion: "sibling-micro-universe-v1";
  targetFamilyKey: typeof TARGET_FAMILY_KEY;
  note: string;
  siblingMicroBucketKeys: string[];
  inclusionRationale: string[];
  perMicroBucketSample: SiblingMicroBucketSampleRow[];
  /** Microedge só sobre episódios paper cujo microBucketKey ∈ siblingMicroBucketKeys. */
  microEdgeAssessment: MicroEdgeAssessmentDigest;
  comparisonWithMainUniverse: SiblingMicroUniverseComparisonDigest;
  thresholdsUsed: Record<string, number | string>;
}

function edgeSummary(a: MicroEdgeAssessmentDigest): SiblingMicroUniverseComparisonDigest["mainUniverseMicroEdgeSummary"] {
  return {
    microEdgeReadVerdict: a.microEdgeReadVerdict,
    eligibleClosedEpisodesForMicroRead: a.eligibleClosedEpisodesForMicroRead,
    microPositiveRate: a.microPositiveRate,
    microNegativeRate: a.microNegativeRate,
    microNeutralEpisodes: a.microNeutralEpisodes,
    repeatedNeutralPattern: a.repeatedNeutralPattern,
  };
}

function computeSiblingPool(args: {
  catalogRow: FamilyPocketRow | undefined;
  econ: PocketEconomicsDigest;
  exec: PocketExecutionDigest;
  minEligible: number;
  maxKeys: number;
  whitelist: Set<string> | null;
}): { keys: string[]; inclusionRationale: string[] } {
  const rationale: string[] = [];
  const stabCatalog = new Set(args.catalogRow?.pocketStabilityKey ?? []);
  const stabEcon = new Set(args.econ.stableMicroBuckets);
  const stabExec = new Set(args.exec.stableMicroKeysAtScan);
  const repeatedEcon = new Set(args.econ.repeatedMicroBuckets);

  const econElig = new Map(args.econ.currentCycle.buckets.map(b => [b.microBucketKey, b.eligibleCount]));
  const execCand = new Map(args.exec.candidateExecutionPockets.map(c => [c.microBucketKey, c]));

  const passesEligible = (k: string): boolean => {
    const ec = econElig.get(k) ?? 0;
    if (ec < args.minEligible) return false;
    const c = execCand.get(k);
    if (c != null && c.eligibleMarketCount < args.minEligible) return false;
    return true;
  };

  const triple = Array.from(stabCatalog).filter(k => stabEcon.has(k) && stabExec.has(k)).sort();
  let pool: string[];
  if (triple.length > 0) {
    pool = triple;
    rationale.push(
      `Interseção tripla (catalog.pocketStabilityKey ∩ economics.stableMicroBuckets ∩ execution.stableMicroKeysAtScan): ${triple.length} chave(s) antes de filtros de elegibilidade.`,
    );
  } else {
    pool = Array.from(stabEcon).filter(k => stabExec.has(k) && (repeatedEcon.has(k) || stabCatalog.has(k))).sort();
    rationale.push(
      "Interseção tripla vazia — fallback conservador: economics.stableMicroBuckets ∩ execution.stableMicroKeysAtScan, restringido a (economics.repeatedMicroBuckets OU catalog.pocketStabilityKey).",
    );
  }

  const afterElig = pool.filter(passesEligible);
  rationale.push(
    `Após exigir eligibleCount≥${args.minEligible} em economics e, se existir candidato execution, eligibleMarketCount≥${args.minEligible}: ${afterElig.length} chave(s).`,
  );
  pool = afterElig;

  if (args.whitelist) {
    const before = pool.length;
    pool = pool.filter(k => args.whitelist!.has(k));
    rationale.push(`Filtro SIBLING_MICRO_BUCKET_WHITELIST: ${before} → ${pool.length} chave(s).`);
  }

  pool.sort((a, b) => {
    const ea = econElig.get(a) ?? 0;
    const eb = econElig.get(b) ?? 0;
    if (eb !== ea) return eb - ea;
    return a.localeCompare(b);
  });

  if (pool.length > args.maxKeys) {
    rationale.push(`Limite SIBLING_MICRO_UNIVERSE_MAX_KEYS=${args.maxKeys} aplicado.`);
    pool = pool.slice(0, args.maxKeys);
  }

  return { keys: pool, inclusionRationale: rationale };
}

function perBucketRows(
  keys: string[],
  catalogRow: FamilyPocketRow | undefined,
  econ: PocketEconomicsDigest,
  exec: PocketExecutionDigest,
): SiblingMicroBucketSampleRow[] {
  const stabCatalog = new Set(catalogRow?.pocketStabilityKey ?? []);
  const stabEcon = new Set(econ.stableMicroBuckets);
  const repeatedEcon = new Set(econ.repeatedMicroBuckets);
  const stabExec = new Set(exec.stableMicroKeysAtScan);
  const econElig = new Map(econ.currentCycle.buckets.map(b => [b.microBucketKey, b.eligibleCount]));
  const execCand = new Map(exec.candidateExecutionPockets.map(c => [c.microBucketKey, c]));

  return keys.map(microBucketKey => {
    const c = execCand.get(microBucketKey);
    return {
      microBucketKey,
      inCatalogPocketStabilityKey: stabCatalog.has(microBucketKey),
      inEconomicsStableMicroBuckets: stabEcon.has(microBucketKey),
      inEconomicsRepeatedMicroBuckets: repeatedEcon.has(microBucketKey),
      eligibleCountEconomics: econElig.has(microBucketKey) ? econElig.get(microBucketKey)! : null,
      inExecutionStableMicroKeysAtScan: stabExec.has(microBucketKey),
      eligibleMarketCountExecution: c?.eligibleMarketCount ?? null,
      betweenSnapshotsStableExecution: c?.betweenSnapshotsStable ?? null,
    };
  });
}

function filterPaperByKeys(entries: readonly MinimalPaperEntry[], keySet: Set<string>): MinimalPaperEntry[] {
  return entries.filter(e => keySet.has(e.microBucketKey)).map(e => JSON.parse(JSON.stringify(e)) as MinimalPaperEntry);
}

function buildComparison(
  mainEdge: MicroEdgeAssessmentDigest,
  sibEdge: MicroEdgeAssessmentDigest,
): SiblingMicroUniverseComparisonDigest {
  const posDelta =
    mainEdge.microPositiveRate != null && sibEdge.microPositiveRate != null
      ? r4(sibEdge.microPositiveRate - mainEdge.microPositiveRate)
      : null;
  const negDelta =
    mainEdge.microNegativeRate != null && sibEdge.microNegativeRate != null
      ? r4(sibEdge.microNegativeRate - mainEdge.microNegativeRate)
      : null;
  const eligDelta = sibEdge.eligibleClosedEpisodesForMicroRead - mainEdge.eligibleClosedEpisodesForMicroRead;

  const interpretationNotes = [
    "O universo principal agrega todos os episódios paper persistidos; o irmão restringe aos microbuckets listados em siblingMicroBucketKeys.",
    "Δ microPositiveRate > 0 sugere que a fatia recorrente/estável concentra mais episódios classificados como micro_positive do que o conjunto global — não implica lucro nem valida edge até prova à parte.",
    "Com amostra pequena no irmão, microEdgeReadVerdict pode ser insufficient_sample mesmo quando o principal já tem corpo estatístico.",
  ];

  return {
    mainUniverseMicroEdgeSummary: edgeSummary(mainEdge),
    siblingUniverseMicroEdgeSummary: edgeSummary(sibEdge),
    deltas: {
      microPositiveRateDelta: posDelta,
      microNegativeRateDelta: negDelta,
      eligibleClosedForMicroReadDelta: eligDelta,
    },
    interpretationNotes,
  };
}

export function buildSiblingMicroUniverseDigest(options?: {
  reusePocketEconomicsDigest?: PocketEconomicsDigest;
  reusePocketExecutionDigest?: PocketExecutionDigest;
}): SiblingMicroUniverseDigest {
  const minEligible = Math.max(1, Math.floor(envNum("SIBLING_MICRO_MIN_ELIGIBLE", 2)));
  const maxKeys = Math.max(1, Math.floor(envNum("SIBLING_MICRO_UNIVERSE_MAX_KEYS", 12)));
  const whitelist = parseWhitelist();

  const catalog = buildCatalogPocketDigest();
  const econ = options?.reusePocketEconomicsDigest ?? buildPocketEconomicsDigest();
  const exec = options?.reusePocketExecutionDigest ?? buildPocketExecutionDigest();

  const catalogRow = catalog.familyRows.find(r => r.familyKey === TARGET_FAMILY_KEY);

  const { keys, inclusionRationale } = computeSiblingPool({
    catalogRow,
    econ,
    exec,
    minEligible,
    maxKeys,
    whitelist,
  });

  const keySet = new Set(keys);
  const allEntries = getMinimalPaperExecutionEntriesReadonly();
  const siblingEntries = filterPaperByKeys(allEntries, keySet);

  const mainEdge = buildMicroEdgeAssessmentFromEntries([...allEntries]);
  const siblingEdge = buildMicroEdgeAssessmentFromEntries(siblingEntries);

  const thresholdsUsed: Record<string, number | string> = {
    SIBLING_MICRO_MIN_ELIGIBLE: minEligible,
    SIBLING_MICRO_UNIVERSE_MAX_KEYS: maxKeys,
    SIBLING_MICRO_BUCKET_WHITELIST: whitelist ? Array.from(whitelist).join("|") : "(none)",
  };

  return {
    computedAt: new Date().toISOString(),
    probeVersion: "sibling-micro-universe-v1",
    targetFamilyKey: TARGET_FAMILY_KEY,
    note:
      "Derivado só de digests existentes (catalog-pocket, pocket-economics, pocket-execution) e entradas paper já persistidas. Não altera o universo principal nem estados JSON existentes. microEdgeAssessment irmão = mesma função conservadora do principal aplicada ao subconjunto de microbuckets.",
    siblingMicroBucketKeys: keys,
    inclusionRationale,
    perMicroBucketSample: perBucketRows(keys, catalogRow, econ, exec),
    microEdgeAssessment: siblingEdge,
    comparisonWithMainUniverse: buildComparison(mainEdge, siblingEdge),
    thresholdsUsed,
  };
}
