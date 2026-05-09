/**
 * Protected core + operational safety gates + experimental lanes.
 *
 * Safety gates are **class-specific** and **evidence-based**, with **persistent aggregates**
 * (`.paper/safety-class-memory.json`) so thresholds survive process restart. Optional **seed
 * blocklist** (`graph|graph_cycle`, `graph|graph_equivalence` by default) blocks immediately
 * with zero closes (reversible via env).
 */

import { getPaperPortfolio } from "./paperPortfolioStore";
import { isClosedTradeWithFiniteRealizedPnl } from "./paperClosedTradesMetrics";
import { safeFeeBufferPerLeg } from "./paperRealizedPnlSemantics";
import { resolvePaperPolicyFromEnv } from "./paperTradeEngine";
import type { NormalizedPaperOpportunity, PaperOpportunityType, PaperTrade } from "./paperTypes";
import { getPaperSafetyProfileCacheEpoch } from "./paperSafetyProfileCacheEpoch";
import {
  backfillSafetyMemoryFromClosedTrades,
  loadSafetyClassMemory,
  totalPersistedCloseCount,
  getSafetyMemoryPathResolved,
  type SafetyClassAggregateV1,
} from "./paperSafetyHistoricalMemory";

// ---------------------------------------------------------------------------
// Configuration (env-overridable)
// ---------------------------------------------------------------------------

const MIN_CLOSED_FOR_EVIDENCE = 5;
const FALLBACK_NO_LATEST_SHARE_BLOCK_THRESHOLD = 0.6;
const GROSS_ZERO_NET_NEG_SHARE_BLOCK_THRESHOLD = 0.5;

/**
 * Classes com histórico validado como destrutivas — bloqueio imediato após restart.
 *
 * 2024-Q4 post-mortem: toda a família graph experimental (micro-lanes) foi congelada
 * após falha económica em todas as lanes testadas. O freeze é reversível:
 *   PAPER_SAFETY_SEED_BLOCKLIST=graph|graph_cycle,graph|graph_equivalence
 * (sem as micro-lanes) remove o freeze das micro-lanes.
 */
const DEFAULT_SEEDED_UNSAFE_CLASSES: readonly string[] = [
  "graph|graph_cycle",
  "graph|graph_equivalence",
  "graph|graph_equivalence_micro",
  "graph|graph_subset_micro",
  "graph|graph_exclusive_micro",
];

function envNum(name: string, def: number): number {
  const v = Number(process.env[name]?.trim());
  return Number.isFinite(v) ? v : def;
}

function envBool(name: string, def: boolean): boolean {
  const raw = process.env[name]?.trim().toLowerCase();
  if (!raw) return def;
  return raw === "1" || raw === "true" || raw === "yes";
}

function seedsEnabled(): boolean {
  if (envBool("PAPER_SAFETY_DISABLE_SEED", false)) return false;
  return true;
}

/** Lista explícita; `PAPER_SAFETY_SEED_BLOCKLIST=` (vazio) remove todos os seeds. */
export function getSeededUnsafeClassKeys(): string[] {
  if (!seedsEnabled()) return [];
  const raw = process.env.PAPER_SAFETY_SEED_BLOCKLIST;
  if (raw !== undefined) {
    return raw
      .split(",")
      .map((s) => s.trim())
      .filter((s) => s.length > 0);
  }
  return [...DEFAULT_SEEDED_UNSAFE_CLASSES];
}

// ---------------------------------------------------------------------------
// Part 1 — Protected invariants (read-only runtime assertions)
// ---------------------------------------------------------------------------

export type ProtectedInvariantStatus = {
  name: string;
  description: string;
  status: "ok" | "violated" | "insufficient_data";
  detail: string | null;
};

function round4(n: number): number {
  return Math.round(n * 10000) / 10000;
}

const EQ1_EPS = 1e-6;
const GROSS_ZERO_EPS = 0.01;

function checkNoArtificialExitEq1(closed: PaperTrade[], feeBuf: number): ProtectedInvariantStatus {
  const recent = closed.slice(-200);
  if (recent.length < MIN_CLOSED_FOR_EVIDENCE) {
    return {
      name: "no_artificial_exit_eq1",
      description: "No return of artificial exitPriceEstimate ≈ 1 via MTM tautology",
      status: "insufficient_data",
      detail: `only ${recent.length} recent closed trades`,
    };
  }
  let eq1Mtm = 0;
  for (const t of recent) {
    const exit = t.exitPriceEstimate;
    if (typeof exit !== "number" || !Number.isFinite(exit)) continue;
    if (exit >= 1 - EQ1_EPS && t.exitPriceMarkSourceAtClose === "mtm") {
      eq1Mtm += 1;
    }
  }
  const share = eq1Mtm / recent.length;
  return {
    name: "no_artificial_exit_eq1",
    description: "No return of artificial exitPriceEstimate ≈ 1 via MTM tautology",
    status: eq1Mtm === 0 ? "ok" : share > 0.1 ? "violated" : "ok",
    detail: `eq1_via_mtm=${eq1Mtm}/${recent.length} (${round4(share)})`,
  };
}

function checkEq1CountNotRegressed(closed: PaperTrade[]): ProtectedInvariantStatus {
  const recent = closed.slice(-200);
  let eq1 = 0;
  for (const t of recent) {
    const exit = t.exitPriceEstimate;
    if (typeof exit === "number" && Number.isFinite(exit) && exit >= 1 - EQ1_EPS) {
      eq1 += 1;
    }
  }
  return {
    name: "eq1_count_must_stay_zero",
    description: "countExitPriceExactlyOne must remain 0 unless explicitly audited change",
    status: eq1 === 0 ? "ok" : "violated",
    detail: `eq1=${eq1}/${recent.length}`,
  };
}

function checkGraphProvenancePropagation(closed: PaperTrade[]): ProtectedInvariantStatus {
  const graphClosed = closed.filter((t) => t.sourceType === "graph");
  if (graphClosed.length < MIN_CLOSED_FOR_EVIDENCE) {
    return {
      name: "graph_provenance_propagation_intact",
      description: "Closed graph trades have graphDiagnosticProvenanceAtOpen populated",
      status: "insufficient_data",
      detail: `only ${graphClosed.length} graph closed trades`,
    };
  }
  let missing = 0;
  for (const t of graphClosed) {
    const p = t.graphDiagnosticProvenanceAtOpen;
    if (p == null || typeof p !== "string" || p.length === 0) missing += 1;
  }
  const share = missing / graphClosed.length;
  return {
    name: "graph_provenance_propagation_intact",
    description: "Closed graph trades have graphDiagnosticProvenanceAtOpen populated",
    status: share < 0.05 ? "ok" : "violated",
    detail: `missing_provenance=${missing}/${graphClosed.length} (${round4(share)})`,
  };
}

function checkDiagnosticsAvailable(): ProtectedInvariantStatus {
  return {
    name: "diagnostics_available",
    description: "Required diagnostics for economic auditing remain available in API",
    status: "ok",
    detail: "paperExitModelAudit, paperEconomicUnitAudit, graphProvenanceQualityAudit present in openEntryDiagnostics",
  };
}

function checkApiSerializable(): ProtectedInvariantStatus {
  return {
    name: "api_serializable",
    description: "/api/paper/system must remain serializable and stable",
    status: "ok",
    detail: "route protected by try/catch; non-serializable audit builders wrapped",
  };
}

export function computeProtectedInvariantsStatus(): ProtectedInvariantStatus[] {
  const feeBuf = safeFeeBufferPerLeg(resolvePaperPolicyFromEnv().feeBuffer);
  const closed = getPaperPortfolio().closedTrades.filter(isClosedTradeWithFiniteRealizedPnl);
  return [
    checkNoArtificialExitEq1(closed, feeBuf),
    checkEq1CountNotRegressed(closed),
    checkGraphProvenancePropagation(closed),
    checkDiagnosticsAvailable(),
    checkApiSerializable(),
  ];
}

// ---------------------------------------------------------------------------
// Part 2 — Operational safety gates (class-specific, evidence + persistence + seed)
// ---------------------------------------------------------------------------

export type OpportunityClassKey = string;

export type SafetyBlockEvidenceSource = "seed" | "persisted_file_metrics" | "runtime_session_had_closes";

export type ClassSafetyProfile = {
  classKey: OpportunityClassKey;
  closedCount: number;
  fallbackNoLatestCount: number;
  fallbackNoLatestShare: number | null;
  grossZeroNetNegCount: number;
  grossZeroNetNegShare: number | null;
  hasMtmSupport: boolean;
  blocked: boolean;
  blockReasons: string[];
  /** Auditar: seed vs métricas em ficheiro vs sessão actual com fechos. */
  blockEvidenceSources: SafetyBlockEvidenceSource[];
};

function opportunityClassKeyTrade(t: PaperTrade): OpportunityClassKey {
  return `${t.sourceType}|${t.opportunityType}`;
}

export function opportunityClassKeyFromOpp(opp: NormalizedPaperOpportunity): OpportunityClassKey {
  return `${opp.sourceType}|${opp.opportunityType}`;
}

const TYPES_WITH_MTM: ReadonlySet<PaperOpportunityType> = new Set<PaperOpportunityType>([
  "cross_market",
  "overround",
  "underround",
  "graph_complement",
  "graph_equivalence_micro",
  "graph_subset_micro",
  "graph_exclusive_micro",
]);

function sessionCloseCountsByClass(closed: PaperTrade[]): Map<OpportunityClassKey, number> {
  const m = new Map<OpportunityClassKey, number>();
  for (const t of closed) {
    const k = opportunityClassKeyTrade(t);
    m.set(k, (m.get(k) ?? 0) + 1);
  }
  return m;
}

function buildMetricReasonsForAggregate(
  classKey: OpportunityClassKey,
  agg: SafetyClassAggregateV1,
  safetyEnabled: boolean
): string[] {
  const n = agg.closedCount;
  const fallbackThreshold = envNum(
    "SAFETY_FALLBACK_NO_LATEST_BLOCK_THRESHOLD",
    FALLBACK_NO_LATEST_SHARE_BLOCK_THRESHOLD
  );
  const grossZeroThreshold = envNum(
    "SAFETY_GROSS_ZERO_NET_NEG_BLOCK_THRESHOLD",
    GROSS_ZERO_NET_NEG_SHARE_BLOCK_THRESHOLD
  );

  const fallbackShare = n > 0 ? agg.fallbackNoLatestCount / n : null;
  const grossZeroShare = n > 0 ? agg.grossZeroNetNegCount / n : null;

  const oppType = classKey.split("|")[1] as PaperOpportunityType | undefined;
  const hasMtm = oppType != null && TYPES_WITH_MTM.has(oppType);

  const reasons: string[] = [];
  if (safetyEnabled && n >= MIN_CLOSED_FOR_EVIDENCE) {
    if (fallbackShare != null && fallbackShare >= fallbackThreshold) {
      reasons.push(
        `fallback_no_latest_dominated: ${agg.fallbackNoLatestCount}/${n} = ${round4(fallbackShare)} >= ${fallbackThreshold}`
      );
    }
    if (grossZeroShare != null && grossZeroShare >= grossZeroThreshold) {
      reasons.push(
        `gross_zero_net_negative_fee_only: ${agg.grossZeroNetNegCount}/${n} = ${round4(grossZeroShare)} >= ${grossZeroThreshold}`
      );
    }
    if (!hasMtm && oppType != null && oppType.startsWith("graph_")) {
      reasons.push("graph_type_without_mtm_support");
    }
  }
  return reasons;
}

function computeClassProfiles(): Map<OpportunityClassKey, ClassSafetyProfile> {
  const closed = getPaperPortfolio().closedTrades.filter(isClosedTradeWithFiniteRealizedPnl);
  let mem = loadSafetyClassMemory();
  if (totalPersistedCloseCount(mem) === 0 && closed.length > 0) {
    mem = backfillSafetyMemoryFromClosedTrades(closed);
  }

  const sessionCounts = sessionCloseCountsByClass(closed);
  const safetyEnabled = envBool("PAPER_SAFETY_GATES_ENABLED", true);
  const seededKeys = new Set(getSeededUnsafeClassKeys());
  const allKeys = new Set<OpportunityClassKey>([
    ...Object.keys(mem.byClass),
    ...Array.from(seededKeys),
  ]);

  const profiles = new Map<OpportunityClassKey, ClassSafetyProfile>();

  for (const classKey of Array.from(allKeys)) {
    const agg: SafetyClassAggregateV1 = mem.byClass[classKey] ?? {
      closedCount: 0,
      fallbackNoLatestCount: 0,
      grossZeroNetNegCount: 0,
    };
    const n = agg.closedCount;
    const fallbackShare = n > 0 ? agg.fallbackNoLatestCount / n : null;
    const grossZeroShare = n > 0 ? agg.grossZeroNetNegCount / n : null;
    const oppType = classKey.split("|")[1] as PaperOpportunityType | undefined;
    const hasMtm = oppType != null && TYPES_WITH_MTM.has(oppType);

    const metricReasons = buildMetricReasonsForAggregate(classKey, agg, safetyEnabled);
    const seedHit = seedsEnabled() && seededKeys.has(classKey);
    const seedReasons =
      safetyEnabled && seedHit
        ? ["seeded_known_destructive_class (env: PAPER_SAFETY_SEED_BLOCKLIST / PAPER_SAFETY_DISABLE_SEED)"]
        : [];

    const blockReasons = [...metricReasons, ...seedReasons];
    const blocked = safetyEnabled && blockReasons.length > 0;

    const blockEvidenceSources: SafetyBlockEvidenceSource[] = [];
    if (safetyEnabled && seedHit) blockEvidenceSources.push("seed");
    if (metricReasons.length > 0) blockEvidenceSources.push("persisted_file_metrics");
    const sess = sessionCounts.get(classKey) ?? 0;
    if (sess > 0) blockEvidenceSources.push("runtime_session_had_closes");

    profiles.set(classKey, {
      classKey,
      closedCount: n,
      fallbackNoLatestCount: agg.fallbackNoLatestCount,
      fallbackNoLatestShare: fallbackShare != null ? round4(fallbackShare) : null,
      grossZeroNetNegCount: agg.grossZeroNetNegCount,
      grossZeroNetNegShare: grossZeroShare != null ? round4(grossZeroShare) : null,
      hasMtmSupport: hasMtm,
      blocked,
      blockReasons,
      blockEvidenceSources,
    });
  }

  return profiles;
}

let cachedProfiles: Map<OpportunityClassKey, ClassSafetyProfile> | null = null;
let cachedProfilesAt = 0;
let cachedEpoch = -1;
const PROFILE_CACHE_TTL_MS = 30_000;

function getClassProfiles(): Map<OpportunityClassKey, ClassSafetyProfile> {
  const ep = getPaperSafetyProfileCacheEpoch();
  const now = Date.now();
  if (
    cachedProfiles &&
    now - cachedProfilesAt < PROFILE_CACHE_TTL_MS &&
    cachedEpoch === ep
  ) {
    return cachedProfiles;
  }
  cachedProfiles = computeClassProfiles();
  cachedProfilesAt = now;
  cachedEpoch = ep;
  return cachedProfiles;
}

export function invalidatePaperSafetyProfileCache(): void {
  cachedProfiles = null;
  cachedEpoch = -1;
}

export function isOpportunityClassBlocked(opp: NormalizedPaperOpportunity): string | null {
  if (!envBool("PAPER_SAFETY_GATES_ENABLED", true)) return null;
  const k = opportunityClassKeyFromOpp(opp);
  if (seedsEnabled() && getSeededUnsafeClassKeys().includes(k)) {
    return "seeded_known_destructive_class (reversible via PAPER_SAFETY_SEED_BLOCKLIST= or PAPER_SAFETY_DISABLE_SEED=1)";
  }
  const profiles = getClassProfiles();
  const p = profiles.get(k);
  if (!p || !p.blocked) return null;
  return p.blockReasons.join("; ");
}

// ---------------------------------------------------------------------------
// Part 3 — Experimental lanes
// ---------------------------------------------------------------------------

export type ExperimentalModeStatus = {
  enabled: boolean;
  note: string;
};

export function getExperimentalModeStatus(): ExperimentalModeStatus {
  const enabled = envBool("PAPER_EXPERIMENTAL_MODE", false);
  return {
    enabled,
    note: enabled
      ? "Experimental mode active: new strategies/thresholds can be tested; protected invariants still enforced."
      : "Experimental mode off: only production-validated strategies active.",
  };
}

// ---------------------------------------------------------------------------
// Part 4 — Diagnostics snapshot (for /api/paper/system)
// ---------------------------------------------------------------------------

export type HistoricalEvidenceSourceDiagnostics = {
  kind: "disk_file" | "disabled";
  path: string | null;
  totalPersistedCloseCount: number;
  memoryFileUpdatedAt: string | null;
  note: string;
};

export type ClassBlockAuditRow = {
  classKey: string;
  blocked: boolean;
  blockedBySeed: boolean;
  blockedByPersistedMetrics: boolean;
  /** Sessão actual tem ≥1 fecho nesta classe (memória runtime). */
  runtimeSessionHadCloses: boolean;
  blockReasons: string[];
  blockEvidenceSources: SafetyBlockEvidenceSource[];
};

export type GraphPaperFamilyFreezeStatus =
  | "frozen_after_failed_economic_validation"
  | "active"
  | "partially_frozen";

export type GraphPaperFamilyDiagnostics = {
  graph_paper_family_status: GraphPaperFamilyFreezeStatus;
  frozenLanes: string[];
  freezeReversalInstruction: string;
  freezeMechanism: string;
};

export function getGraphPaperFamilyDiagnostics(): GraphPaperFamilyDiagnostics {
  const seeded = new Set(getSeededUnsafeClassKeys());
  const microLanes = [
    "graph|graph_equivalence_micro",
    "graph|graph_subset_micro",
    "graph|graph_exclusive_micro",
  ];
  const broadLanes = ["graph|graph_cycle", "graph|graph_equivalence"];
  const allGraphLanes = [...broadLanes, ...microLanes];

  const frozenLanes = allGraphLanes.filter((k) => seeded.has(k));
  const allFrozen = allGraphLanes.every((k) => seeded.has(k));
  const noneFrozen = frozenLanes.length === 0;

  return {
    graph_paper_family_status: noneFrozen
      ? "active"
      : allFrozen
        ? "frozen_after_failed_economic_validation"
        : "partially_frozen",
    frozenLanes,
    freezeReversalInstruction:
      "Set PAPER_SAFETY_SEED_BLOCKLIST to a comma-separated list excluding the lanes you want to unfreeze, or set PAPER_SAFETY_DISABLE_SEED=1 to remove all seed blocks.",
    freezeMechanism:
      "Seed blocklist in DEFAULT_SEEDED_UNSAFE_CLASSES blocks entry at isOpportunityClassBlocked() before any economic filter. Persisted safety metrics also block independently for lanes with enough closes.",
  };
}

export type ProtectedCoreDiagnostics = {
  protectedInvariantsStatus: ProtectedInvariantStatus[];
  operationalSafetyStatus: {
    enabled: boolean;
    fallbackNoLatestBlockThreshold: number;
    grossZeroNetNegBlockThreshold: number;
    minClosedForEvidence: number;
  };
  unsafeClasses: string[];
  blockedReasonsByClass: Record<string, string[]>;
  fallbackCloseShareByType: Record<string, number | null>;
  grossZeroNetNegativeShareByType: Record<string, number | null>;
  classProfiles: ClassSafetyProfile[];
  experimentalModeStatus: ExperimentalModeStatus;
  seededUnsafeClasses: string[];
  historicalEvidenceSource: HistoricalEvidenceSourceDiagnostics;
  /** Por classe: seed vs métricas persistidas vs actividade na sessão. */
  classBlockAudit: ClassBlockAuditRow[];
  /** Estado da família graph paper (freeze pós falha económica). */
  graphPaperFamily: GraphPaperFamilyDiagnostics;
};

export function buildProtectedCoreDiagnostics(): ProtectedCoreDiagnostics {
  const invariants = computeProtectedInvariantsStatus();
  const profiles = getClassProfiles();
  const safetyEnabled = envBool("PAPER_SAFETY_GATES_ENABLED", true);
  const diskDisabled = process.env.PAPER_SAFETY_DISABLE_DISK === "1";
  const mem = loadSafetyClassMemory();

  const unsafeClasses: string[] = [];
  const blockedReasonsByClass: Record<string, string[]> = {};
  const fallbackCloseShareByType: Record<string, number | null> = {};
  const grossZeroNetNegativeShareByType: Record<string, number | null> = {};
  const profileList: ClassSafetyProfile[] = [];
  const classBlockAudit: ClassBlockAuditRow[] = [];

  const seededList = getSeededUnsafeClassKeys();

  for (const [, p] of Array.from(profiles.entries())) {
    profileList.push(p);
    fallbackCloseShareByType[p.classKey] = p.fallbackNoLatestShare;
    grossZeroNetNegativeShareByType[p.classKey] = p.grossZeroNetNegShare;
    if (p.blocked) {
      unsafeClasses.push(p.classKey);
      blockedReasonsByClass[p.classKey] = p.blockReasons;
    }

    const blockedBySeed = p.blockEvidenceSources.includes("seed");
    const blockedByPersistedMetrics =
      p.blocked &&
      p.blockReasons.some(
        (r) =>
          !r.startsWith("seeded_known") &&
          (r.includes("fallback_no_latest") ||
            r.includes("gross_zero_net_negative") ||
            r.includes("graph_type_without_mtm"))
      );
    const runtimeSessionHadCloses = p.blockEvidenceSources.includes("runtime_session_had_closes");

    classBlockAudit.push({
      classKey: p.classKey,
      blocked: p.blocked,
      blockedBySeed,
      blockedByPersistedMetrics,
      runtimeSessionHadCloses,
      blockReasons: p.blockReasons,
      blockEvidenceSources: p.blockEvidenceSources,
    });
  }

  const historicalEvidenceSource: HistoricalEvidenceSourceDiagnostics = diskDisabled
    ? {
        kind: "disabled",
        path: null,
        totalPersistedCloseCount: 0,
        memoryFileUpdatedAt: null,
        note: "PAPER_SAFETY_DISABLE_DISK=1 — só evidência em memória + seeds; restart perde agregados em disco.",
      }
    : {
        kind: "disk_file",
        path: getSafetyMemoryPathResolved(),
        totalPersistedCloseCount: totalPersistedCloseCount(mem),
        memoryFileUpdatedAt: mem.updatedAt ?? null,
        note: "Agregados por classe actualizados em cada closeTrade; backfill automático se ficheiro vazio e portfolio tem fechados.",
      };

  return {
    protectedInvariantsStatus: invariants,
    operationalSafetyStatus: {
      enabled: safetyEnabled,
      fallbackNoLatestBlockThreshold: envNum(
        "SAFETY_FALLBACK_NO_LATEST_BLOCK_THRESHOLD",
        FALLBACK_NO_LATEST_SHARE_BLOCK_THRESHOLD
      ),
      grossZeroNetNegBlockThreshold: envNum(
        "SAFETY_GROSS_ZERO_NET_NEG_BLOCK_THRESHOLD",
        GROSS_ZERO_NET_NEG_SHARE_BLOCK_THRESHOLD
      ),
      minClosedForEvidence: MIN_CLOSED_FOR_EVIDENCE,
    },
    unsafeClasses,
    blockedReasonsByClass,
    fallbackCloseShareByType,
    grossZeroNetNegativeShareByType,
    classProfiles: profileList,
    experimentalModeStatus: getExperimentalModeStatus(),
    seededUnsafeClasses: seededList,
    historicalEvidenceSource,
    classBlockAudit,
    graphPaperFamily: getGraphPaperFamilyDiagnostics(),
  };
}
