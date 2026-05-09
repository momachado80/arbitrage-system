/**
 * Pocket Economics Probe — observabilidade da explorabilidade económica de um pocket
 * recorrente na família fixa other:price_above:>3m.
 *
 * NÃO mede PnL, NÃO valida edge, NÃO simula lucro. Heurísticas de plausibilidade
 * de substrato apenas. Isolado em globalThis; não altera catalog-pocket.
 */

import fs from "fs";
import path from "path";
import { defaultPaperTrailStatePath, PAPER_TRAIL_FILENAMES } from "./paperStateDir";

function envNum(k: string, d: number): number {
  const v = Number(process.env[k]?.trim());
  return Number.isFinite(v) ? v : d;
}

function envStr(k: string, d: string): string {
  const v = process.env[k]?.trim();
  return v && v.length > 0 ? v : d;
}

const GAMMA_URL = "https://gamma-api.polymarket.com/markets";
const GAMMA_FETCH_TIMEOUT_MS = 15_000;
const SCAN_INTERVAL_MS = () => envNum("POCKET_ECON_SCAN_INTERVAL_MS", 6 * 3600_000);

const ELIGIBLE_MAX_SPREAD = 0.15;
const ELIGIBLE_MIN_LIQ = 500;
const ELIGIBLE_MIN_PRICE = 0.03;
const ELIGIBLE_MAX_PRICE = 0.97;
const MICRO_BUCKET_KEY_MAX_LEN = () => envNum("POCKET_ECON_KEY_MAX_LEN", 192);
const STABLE_SCANS = () => envNum("POCKET_ECON_STABLE_SCANS", 3);
const HISTORY_MAX = () => envNum("POCKET_ECON_HISTORY_MAX", 10);
const PERSIST_THROTTLE_MS = () => envNum("POCKET_ECON_PERSIST_THROTTLE_MS", 30_000);
const PERSIST_VERSION = 1 as const;

/** Gates explícitos para “merece probe de execução mínimo?” — só observacional; não é edge nem ordem de trade. */
const PROMO_MIN_ELIGIBLE_HISTORY_SCANS = () =>
  envNum("POCKET_ECON_PROMO_MIN_ELIGIBLE_HISTORY_SCANS", STABLE_SCANS());
const PROMO_MIN_STABLE_POCKET_SHARE = () => envNum("POCKET_ECON_PROMO_MIN_STABLE_POCKET_SHARE", 0.12);
const PROMO_MIN_REPEATED_MICRO_BUCKETS = () =>
  envNum("POCKET_ECON_PROMO_MIN_REPEATED_MICRO_BUCKETS", 2);
const PROMO_MIN_AVG_ELIGIBLE_STABLE = () => envNum("POCKET_ECON_PROMO_MIN_AVG_ELIGIBLE_STABLE", 1.5);
const PROMO_MIN_LARGEST_STABLE_SIZE = () => envNum("POCKET_ECON_PROMO_MIN_LARGEST_STABLE_SIZE", 2);
const PROMO_MIN_PERSISTENCE_SCORE = () => envNum("POCKET_ECON_PROMO_MIN_PERSISTENCE_SCORE", 0.35);

/** Família fechada — não parametrizar universo além disto nesta v1 */
const TARGET_FAMILY_KEY = "other:price_above:>3m" as const;

type Underlying = "BTC" | "ETH" | "SOL" | "political" | "sports" | "other_crypto" | "other";
type Template = "price_above" | "price_below" | "price_range" | "yes_no_event" | "multi_outcome" | "other";
type HorizonBucket = "<24h" | "24h-168h" | "1w-4w" | "1m-3m" | ">3m" | "unknown";

interface GammaRawMarket {
  id: string; question: string; outcomes: string; outcomePrices: string;
  liquidity: string; volume: string; active: boolean; closed: boolean;
  market_slug: string; end_date_iso?: string; category?: string;
}

export interface PocketEconomicsMarketRow {
  id: string;
  question: string;
  liquidity: number;
  spread: number;
  prices: number[];
  outcomes: string[];
}

export interface PocketEconomicsBucketRow {
  microBucketKey: string;
  eligibleCount: number;
  prioritized: boolean;
  marketIds: string[];
  markets: PocketEconomicsMarketRow[];
  spreadProxyMedian: number | null;
  localDensity: number | null;
}

export type PocketEconomicsPersistenceLoadStatus =
  | "ok"
  | "missing"
  | "invalid"
  | "error"
  | "disabled"
  | "pending";

export interface PocketEconomicsPersistenceDigest {
  persistenceEnabled: boolean;
  persistencePath: string | null;
  lastPersistenceWriteAt: string | null;
  lastPersistenceLoadAt: string | null;
  persistenceLoadStatus: PocketEconomicsPersistenceLoadStatus;
}

export type PocketEconomicsPromotionVerdict =
  | "not_ready"
  | "borderline"
  | "ready_for_minimal_execution_probe";

export interface PocketEconomicsPromotionThresholdsDigest {
  minEligibleHistoryScans: number;
  minStablePocketShare: number;
  minRepeatedMicroBuckets: number;
  minAverageEligibleStable: number;
  minLargestStablePocketSize: number;
  minPersistenceScore: number;
}

export interface PocketEconomicsPromotionAssessment {
  minimumStableScansSatisfied: boolean;
  stablePocketShareSatisfied: boolean;
  minimumRepeatedMicroBucketsSatisfied: boolean;
  minimumAverageEligibleCountSatisfied: boolean;
  largestStablePocketSizeSatisfied: boolean;
  persistenceScoreSatisfied: boolean;
  overallPromotionVerdict: PocketEconomicsPromotionVerdict;
  promotionReasons: string[];
  blockingReasons: string[];
  thresholdsUsed: PocketEconomicsPromotionThresholdsDigest;
}

export interface PocketEconomicsDigest {
  computedAt: string;
  probeVersion: "pocket-economics-v1";
  scanStatus: "completed" | "scanning" | "idle" | "error";
  targetFamilyKey: typeof TARGET_FAMILY_KEY;
  note: string;
  prioritizedMicroBucketKeys: string[];
  repeatedMicroBuckets: string[];
  newMicroBuckets: string[];
  droppedMicroBuckets: string[];
  stableMicroBuckets: string[];
  averageEligibleCountInStablePocket: number | null;
  largestStablePocketSize: number;
  stablePocketShare: number | null;
  pocketPersistenceScore: number | null;
  /** 0–1 heurística observacional; não é edge nem PnL */
  recurrentSubstratePlausibilityScore: number | null;
  currentCycle: {
    totalActiveInFamily: number;
    totalEligibleInFamily: number;
    buckets: PocketEconomicsBucketRow[];
  };
  previousScanTimestamp: string | null;
  scanDurationMs: number | null;
  scanTimestamp: string | null;
  totalMarketsScanned: number;
  lastScanStartAt: string | null;
  lastScanEndAt: string | null;
  lastSuccessfulScanAt: string | null;
  lastScanErrorAt: string | null;
  lastScanErrorMessage: string | null;
  isScanRunning: boolean;
  currentRunId: number;
  nextScheduledScanAt: string | null;
  schedulerStartedAt: string | null;
  totalScanAttempts: number;
  totalScanSuccess: number;
  totalScanErrors: number;
  totalScanSkippedBusy: number;
  persistence: PocketEconomicsPersistenceDigest;
  promotionAssessment: PocketEconomicsPromotionAssessment;
}

const GLOBAL_KEY = "__pocketEconomicsProbe_v1";

interface LastMetrics {
  repeatedMicroBuckets: string[];
  newMicroBuckets: string[];
  droppedMicroBuckets: string[];
  stableMicroBuckets: string[];
  averageEligibleCountInStablePocket: number | null;
  largestStablePocketSize: number;
  stablePocketShare: number | null;
  pocketPersistenceScore: number | null;
  recurrentSubstratePlausibilityScore: number | null;
  totalActiveInFamily: number;
  totalEligibleInFamily: number;
}

interface PersistedBucketLite {
  microBucketKey: string;
  eligibleCount: number;
  prioritized: boolean;
  marketIds: string[];
  spreadProxyMedian: number | null;
  localDensity: number | null;
}

interface PocketEconomicsPersistedFileV1 {
  version: typeof PERSIST_VERSION;
  savedAt: string;
  previousEligibleByBucket: Record<string, number>;
  previousScanTs: number | null;
  eligibleHistory: Array<Record<string, number>>;
  lastMetricsSnapshot: LastMetrics | null;
  lastDigestBucketsLite: PersistedBucketLite[];
  currentRunId: number;
  totalScanAttempts: number;
  totalScanSuccess: number;
  totalScanErrors: number;
  totalScanSkippedBusy: number;
  lastScanTimestamp: number | null;
  lastScanDurationMs: number | null;
  totalMarketsScanned: number;
  lastScanStartAt: number | null;
  lastScanEndAt: number | null;
  lastSuccessfulScanAt: number | null;
  lastScanErrorAt: number | null;
  lastScanErrorMessage: string | null;
  scanError: string | null;
}

const PERSIST_META_GLOBAL_KEY = "__pocketEconomicsPersistMeta_v1";

interface PocketEconomicsPersistMeta {
  lastWriteAtMs: number | null;
  lastLoadAtMs: number | null;
  loadStatus: PocketEconomicsPersistenceLoadStatus;
  lastThrottleWallMs: number;
}

function getPersistMeta(): PocketEconomicsPersistMeta {
  const g = globalThis as unknown as Record<string, PocketEconomicsPersistMeta | undefined>;
  if (!g[PERSIST_META_GLOBAL_KEY]) {
    g[PERSIST_META_GLOBAL_KEY] = {
      lastWriteAtMs: null,
      lastLoadAtMs: null,
      loadStatus: "pending",
      lastThrottleWallMs: 0,
    };
  }
  return g[PERSIST_META_GLOBAL_KEY]!;
}

function isPersistenceDiskDisabled(): boolean {
  return process.env.POCKET_ECON_DISABLE_DISK === "1";
}

function defaultPersistencePath(): string {
  const raw = process.env.POCKET_ECON_STATE_PATH?.trim();
  if (raw && raw.length > 0) return path.resolve(raw);
  return defaultPaperTrailStatePath(PAPER_TRAIL_FILENAMES.pocketEconomics);
}

function bucketsToLite(rows: PocketEconomicsBucketRow[]): PersistedBucketLite[] {
  return rows.map(r => ({
    microBucketKey: r.microBucketKey,
    eligibleCount: r.eligibleCount,
    prioritized: r.prioritized,
    marketIds: r.marketIds.slice(0, 400),
    spreadProxyMedian: r.spreadProxyMedian,
    localDensity: r.localDensity,
  }));
}

function liteToBuckets(lite: PersistedBucketLite[]): PocketEconomicsBucketRow[] {
  return lite.map(b => ({
    microBucketKey: b.microBucketKey,
    eligibleCount: b.eligibleCount,
    prioritized: b.prioritized,
    marketIds: b.marketIds,
    markets: [],
    spreadProxyMedian: b.spreadProxyMedian,
    localDensity: b.localDensity,
  }));
}

function sanitizeEligibleMap(raw: unknown): Record<string, number> {
  const out: Record<string, number> = {};
  if (!raw || typeof raw !== "object") return out;
  for (const [k, v] of Object.entries(raw as Record<string, unknown>)) {
    if (typeof k !== "string" || k.length > 512) continue;
    const n = typeof v === "number" ? v : Number(v);
    if (!Number.isFinite(n) || n < 0) continue;
    out[k] = Math.floor(n);
  }
  return out;
}

function sanitizeHistory(raw: unknown): Array<Record<string, number>> {
  if (!Array.isArray(raw)) return [];
  const max = HISTORY_MAX();
  const rows = raw
    .filter((x): x is Record<string, unknown> => x !== null && typeof x === "object" && !Array.isArray(x))
    .map(x => sanitizeEligibleMap(x));
  return rows.slice(-max);
}

function sanitizeLastMetrics(raw: unknown): LastMetrics | null {
  if (!raw || typeof raw !== "object") return null;
  const o = raw as Record<string, unknown>;
  const strArr = (x: unknown) =>
    Array.isArray(x) ? x.filter((s): s is string => typeof s === "string" && s.length < 600) : [];
  const numOrNull = (x: unknown) => {
    const n = typeof x === "number" ? x : Number(x);
    return Number.isFinite(n) ? n : null;
  };
  return {
    repeatedMicroBuckets: strArr(o.repeatedMicroBuckets),
    newMicroBuckets: strArr(o.newMicroBuckets),
    droppedMicroBuckets: strArr(o.droppedMicroBuckets),
    stableMicroBuckets: strArr(o.stableMicroBuckets),
    averageEligibleCountInStablePocket: numOrNull(o.averageEligibleCountInStablePocket),
    largestStablePocketSize: Math.max(0, Math.floor(numOrNull(o.largestStablePocketSize) ?? 0)),
    stablePocketShare: numOrNull(o.stablePocketShare),
    pocketPersistenceScore: numOrNull(o.pocketPersistenceScore),
    recurrentSubstratePlausibilityScore: numOrNull(o.recurrentSubstratePlausibilityScore),
    totalActiveInFamily: Math.max(0, Math.floor(numOrNull(o.totalActiveInFamily) ?? 0)),
    totalEligibleInFamily: Math.max(0, Math.floor(numOrNull(o.totalEligibleInFamily) ?? 0)),
  };
}

function sanitizeBucketsLite(raw: unknown): PersistedBucketLite[] {
  if (!Array.isArray(raw)) return [];
  const out: PersistedBucketLite[] = [];
  for (const item of raw) {
    if (!item || typeof item !== "object") continue;
    const o = item as Record<string, unknown>;
    const k = o.microBucketKey;
    if (typeof k !== "string" || k.length < 1 || k.length > 600) continue;
    const ec = typeof o.eligibleCount === "number" ? o.eligibleCount : Number(o.eligibleCount);
    if (!Number.isFinite(ec)) continue;
    const ids = Array.isArray(o.marketIds)
      ? o.marketIds.filter((id): id is string => typeof id === "string" && id.length < 64).slice(0, 400)
      : [];
    const sm = o.spreadProxyMedian;
    const spreadProxyMedian =
      sm === null || sm === undefined
        ? null
        : Number.isFinite(Number(sm))
          ? Number(sm)
          : null;
    const ld = o.localDensity;
    const localDensity =
      ld === null || ld === undefined
        ? null
        : Number.isFinite(Number(ld))
          ? Number(ld)
          : null;
    out.push({
      microBucketKey: k,
      eligibleCount: Math.max(0, Math.floor(ec)),
      prioritized: o.prioritized === true,
      marketIds: ids,
      spreadProxyMedian,
      localDensity,
    });
  }
  return out;
}

function numOrNullFinite(v: unknown): number | null {
  if (typeof v !== "number" || !Number.isFinite(v)) return null;
  return v;
}

function hydrateFromDisk(st: ProbeState): void {
  const meta = getPersistMeta();
  if (isPersistenceDiskDisabled()) {
    meta.loadStatus = "disabled";
    meta.lastLoadAtMs = Date.now();
    return;
  }
  const p = defaultPersistencePath();
  try {
    if (!fs.existsSync(p)) {
      meta.loadStatus = "missing";
      meta.lastLoadAtMs = Date.now();
      return;
    }
    const rawFile = fs.readFileSync(p, "utf8");
    const j = JSON.parse(rawFile) as Partial<PocketEconomicsPersistedFileV1>;
    if (j.version !== PERSIST_VERSION) {
      meta.loadStatus = "invalid";
      meta.lastLoadAtMs = Date.now();
      return;
    }

    st.previousEligibleByBucket = sanitizeEligibleMap(j.previousEligibleByBucket);
    st.previousScanTs = numOrNullFinite(j.previousScanTs);
    st.eligibleHistory = sanitizeHistory(j.eligibleHistory);
    st.lastMetricsSnapshot = sanitizeLastMetrics(j.lastMetricsSnapshot);
    st.lastDigestBuckets = liteToBuckets(sanitizeBucketsLite(j.lastDigestBucketsLite));

    if (typeof j.currentRunId === "number" && Number.isFinite(j.currentRunId)) {
      st.currentRunId = Math.max(0, Math.floor(j.currentRunId));
    }
    if (typeof j.totalScanAttempts === "number" && Number.isFinite(j.totalScanAttempts)) {
      st.totalScanAttempts = Math.max(0, Math.floor(j.totalScanAttempts));
    }
    if (typeof j.totalScanSuccess === "number" && Number.isFinite(j.totalScanSuccess)) {
      st.totalScanSuccess = Math.max(0, Math.floor(j.totalScanSuccess));
    }
    if (typeof j.totalScanErrors === "number" && Number.isFinite(j.totalScanErrors)) {
      st.totalScanErrors = Math.max(0, Math.floor(j.totalScanErrors));
    }
    if (typeof j.totalScanSkippedBusy === "number" && Number.isFinite(j.totalScanSkippedBusy)) {
      st.totalScanSkippedBusy = Math.max(0, Math.floor(j.totalScanSkippedBusy));
    }

    st.lastScanTimestamp = numOrNullFinite(j.lastScanTimestamp);
    st.lastScanDurationMs = numOrNullFinite(j.lastScanDurationMs);
    if (typeof j.totalMarketsScanned === "number" && Number.isFinite(j.totalMarketsScanned)) {
      st.totalMarketsScanned = Math.max(0, Math.floor(j.totalMarketsScanned));
    }
    st.lastScanStartAt = numOrNullFinite(j.lastScanStartAt);
    st.lastScanEndAt = numOrNullFinite(j.lastScanEndAt);
    st.lastSuccessfulScanAt = numOrNullFinite(j.lastSuccessfulScanAt);
    st.lastScanErrorAt = numOrNullFinite(j.lastScanErrorAt);
    st.lastScanErrorMessage =
      typeof j.lastScanErrorMessage === "string" && j.lastScanErrorMessage.length < 4000
        ? j.lastScanErrorMessage
        : null;
    st.scanError =
      typeof j.scanError === "string" && j.scanError.length < 4000 ? j.scanError : null;

    meta.loadStatus = "ok";
    meta.lastLoadAtMs = Date.now();
    if (typeof j.savedAt === "string") {
      const t = Date.parse(j.savedAt);
      if (Number.isFinite(t)) meta.lastWriteAtMs = t;
    }
    console.log("[PocketEconomics] Rehydrated observational state from disk:", p);
  } catch (e) {
    meta.loadStatus = "error";
    meta.lastLoadAtMs = Date.now();
    console.warn("[PocketEconomics] Persistence load failed (non-fatal):", e instanceof Error ? e.message : e);
  }
}

function buildPersistPayload(st: ProbeState): PocketEconomicsPersistedFileV1 {
  return {
    version: PERSIST_VERSION,
    savedAt: new Date().toISOString(),
    previousEligibleByBucket: { ...st.previousEligibleByBucket },
    previousScanTs: st.previousScanTs,
    eligibleHistory: st.eligibleHistory.map(h => ({ ...h })),
    lastMetricsSnapshot: st.lastMetricsSnapshot ? { ...st.lastMetricsSnapshot } : null,
    lastDigestBucketsLite: bucketsToLite(st.lastDigestBuckets),
    currentRunId: st.currentRunId,
    totalScanAttempts: st.totalScanAttempts,
    totalScanSuccess: st.totalScanSuccess,
    totalScanErrors: st.totalScanErrors,
    totalScanSkippedBusy: st.totalScanSkippedBusy,
    lastScanTimestamp: st.lastScanTimestamp,
    lastScanDurationMs: st.lastScanDurationMs,
    totalMarketsScanned: st.totalMarketsScanned,
    lastScanStartAt: st.lastScanStartAt,
    lastScanEndAt: st.lastScanEndAt,
    lastSuccessfulScanAt: st.lastSuccessfulScanAt,
    lastScanErrorAt: st.lastScanErrorAt,
    lastScanErrorMessage: st.lastScanErrorMessage,
    scanError: st.scanError,
  };
}

function writePersistenceAtomic(filePath: string, payload: PocketEconomicsPersistedFileV1): void {
  const dir = path.dirname(filePath);
  fs.mkdirSync(dir, { recursive: true });
  const tmp = `${filePath}.${process.pid}.${Date.now()}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(payload), "utf8");
  fs.renameSync(tmp, filePath);
}

function maybePersistPocketEconomicsState(st: ProbeState, forceBecauseScanMutated: boolean): void {
  if (isPersistenceDiskDisabled()) return;
  const meta = getPersistMeta();
  const now = Date.now();
  if (!forceBecauseScanMutated && now - meta.lastThrottleWallMs < PERSIST_THROTTLE_MS()) {
    return;
  }
  const filePath = defaultPersistencePath();
  try {
    writePersistenceAtomic(filePath, buildPersistPayload(st));
    meta.lastThrottleWallMs = now;
    meta.lastWriteAtMs = now;
    if (meta.loadStatus === "missing" || meta.loadStatus === "pending") {
      meta.loadStatus = "ok";
    }
  } catch (e) {
    console.warn("[PocketEconomics] Persistence write failed (non-fatal):", e instanceof Error ? e.message : e);
  }
}

function buildPersistenceDigest(): PocketEconomicsPersistenceDigest {
  const disabled = isPersistenceDiskDisabled();
  const meta = getPersistMeta();
  return {
    persistenceEnabled: !disabled,
    persistencePath: disabled ? null : defaultPersistencePath(),
    lastPersistenceWriteAt: meta.lastWriteAtMs ? new Date(meta.lastWriteAtMs).toISOString() : null,
    lastPersistenceLoadAt: meta.lastLoadAtMs ? new Date(meta.lastLoadAtMs).toISOString() : null,
    persistenceLoadStatus: meta.loadStatus,
  };
}

interface ProbeState {
  loopStarted: boolean;
  scanning: boolean;
  scanError: string | null;
  lastScanTimestamp: number | null;
  lastScanDurationMs: number | null;
  totalMarketsScanned: number;
  lastScanStartAt: number | null;
  lastScanEndAt: number | null;
  lastSuccessfulScanAt: number | null;
  lastScanErrorAt: number | null;
  lastScanErrorMessage: string | null;
  currentRunId: number;
  nextScheduledScanAt: number | null;
  schedulerStartedAt: number | null;
  scheduledTimeoutId: ReturnType<typeof setTimeout> | null;
  totalScanAttempts: number;
  totalScanSuccess: number;
  totalScanErrors: number;
  totalScanSkippedBusy: number;
  /** último mapa microBucket -> eligibleCount (scan anterior) */
  previousEligibleByBucket: Record<string, number>;
  previousScanTs: number | null;
  /** histórico recente: cada entrada é microBucket -> eligibleCount */
  eligibleHistory: Array<Record<string, number>>;
  lastDigestBuckets: PocketEconomicsBucketRow[];
  lastMetricsSnapshot: LastMetrics | null;
}

function getState(): ProbeState {
  const g = globalThis as unknown as Record<string, ProbeState | undefined>;
  if (!g[GLOBAL_KEY]) {
    g[GLOBAL_KEY] = {
      loopStarted: false, scanning: false, scanError: null,
      lastScanTimestamp: null, lastScanDurationMs: null, totalMarketsScanned: 0,
      lastScanStartAt: null, lastScanEndAt: null, lastSuccessfulScanAt: null,
      lastScanErrorAt: null, lastScanErrorMessage: null,
      currentRunId: 0, nextScheduledScanAt: null, schedulerStartedAt: null,
      scheduledTimeoutId: null,
      totalScanAttempts: 0, totalScanSuccess: 0, totalScanErrors: 0, totalScanSkippedBusy: 0,
      previousEligibleByBucket: {}, previousScanTs: null,
      eligibleHistory: [], lastDigestBuckets: [], lastMetricsSnapshot: null,
    };
    hydrateFromDisk(g[GLOBAL_KEY]!);
  }
  const st = g[GLOBAL_KEY]!;
  if (!st.eligibleHistory) st.eligibleHistory = [];
  if (!st.lastDigestBuckets) st.lastDigestBuckets = [];
  if (st.previousEligibleByBucket === undefined) st.previousEligibleByBucket = {};
  if (st.lastMetricsSnapshot === undefined) st.lastMetricsSnapshot = null;
  return st;
}

// --- classificação / micro-bucket / elegibilidade: espelho do catalog-pocket (mesmos limiares) ---

const DATE_IN_QUESTION_RE = /\b(?:by|on|before)\s+(?:(\w+ \d{1,2},?\s*\d{4})|(\w+ \d{4})|(\d{4}-\d{2}-\d{2}))\b/i;
const PARSEABLE_MONETARY_RE = /\$\s?[\d,]+(?:\.\d+)?\s*[kKmM]?/gi;

function normalizeQuestionStructuralKey(question: string): string {
  let s = question;
  s = s.replace(/\u200b|\u200c|\u200d|\ufeff/g, "");
  s = s.replace(/[\u2018\u2019]/g, "'");
  s = s.replace(/[\u201c\u201d]/g, '"');
  s = s.toLowerCase();
  s = s.replace(/\s+/g, " ").trim();
  let strikeIdx = 0;
  s = s.replace(PARSEABLE_MONETARY_RE, () => `$_STRIKE${++strikeIdx}`);
  const maxLen = MICRO_BUCKET_KEY_MAX_LEN();
  if (s.length > maxLen) s = s.slice(0, maxLen);
  return s;
}

function classifyUnderlying(question: string, category?: string): Underlying {
  const q = question.toLowerCase();
  const cat = (category ?? "").toLowerCase();
  if (/\b(btc|bitcoin)\b/.test(q)) return "BTC";
  if (/\b(eth|ethereum)\b/.test(q)) return "ETH";
  if (/\bsol(ana)?\b/.test(q)) return "SOL";
  if (/\b(trump|biden|election|president|congress|senate|governor)\b/.test(q) || cat === "politics") return "political";
  if (/\b(nfl|nba|fifa|world cup|nhl|mlb|ufc|premier league|champions league)\b/.test(q) || cat === "sports") return "sports";
  if (/\b(crypto|coin|token|doge|xrp|ada|avax|bnb|link|matic|shib)\b/.test(q) || cat === "crypto") return "other_crypto";
  return "other";
}

function classifyTemplate(question: string, outcomes: string[]): Template {
  if (outcomes.length > 2) return "multi_outcome";
  const q = question.toLowerCase();
  if (/between.*\$.*and.*\$/.test(q)) return "price_range";
  if (/will.*(?:hit|reach|above|exceed|over|surpass).*\$[\d]/.test(q)) return "price_above";
  if (/will.*(?:fall|drop|below|under).*\$[\d]/.test(q)) return "price_below";
  if (outcomes.length === 2) {
    const sorted = outcomes.map(o => o.toLowerCase()).sort();
    if (sorted[0] === "no" && sorted[1] === "yes") return "yes_no_event";
  }
  return "other";
}

function parseEndDate(endDateIso: string | undefined, question: string): Date | null {
  if (endDateIso) {
    const d = new Date(endDateIso);
    if (Number.isFinite(d.getTime())) return d;
  }
  const m = DATE_IN_QUESTION_RE.exec(question);
  if (m) {
    const d = new Date(m[1] ?? m[2] ?? m[3] ?? "");
    if (Number.isFinite(d.getTime())) return d;
  }
  return null;
}

function classifyHorizon(endDate: Date | null, now: number): HorizonBucket {
  if (!endDate) return "unknown";
  const hoursLeft = (endDate.getTime() - now) / 3_600_000;
  if (hoursLeft <= 0) return "<24h";
  if (hoursLeft < 24) return "<24h";
  if (hoursLeft <= 168) return "24h-168h";
  if (hoursLeft <= 672) return "1w-4w";
  if (hoursLeft <= 2160) return "1m-3m";
  return ">3m";
}

function familyKey(u: Underlying, t: Template, h: HorizonBucket): string {
  return `${u}:${t}:${h}`;
}

function isEligible(spread: number, liquidity: number, inPriceBand: boolean): boolean {
  return spread < ELIGIBLE_MAX_SPREAD && liquidity >= ELIGIBLE_MIN_LIQ && inPriceBand;
}

async function fetchGammaPage(offset: number): Promise<GammaRawMarket[]> {
  const url = `${GAMMA_URL}?limit=100&offset=${offset}&active=true&closed=false`;
  const res = await fetch(url, { signal: AbortSignal.timeout(GAMMA_FETCH_TIMEOUT_MS), headers: { Accept: "application/json" } });
  if (!res.ok) throw new Error(`Gamma ${res.status}`);
  return (await res.json()) as GammaRawMarket[];
}

async function fetchAllActive(): Promise<GammaRawMarket[]> {
  const all: GammaRawMarket[] = [];
  for (let p = 0; p < 50; p++) {
    const page = await fetchGammaPage(p * 100);
    all.push(...page);
    if (page.length < 100) break;
    if (p > 0 && p % 5 === 0) {
      await new Promise<void>(r => {
        setImmediate(r);
      });
    }
  }
  return all;
}

function parsePriorityKeys(): string[] {
  const raw = envStr("POCKET_ECON_PRIORITY_MICRO_BUCKETS", "");
  if (!raw) return [];
  return raw.split(/[|,]/).map(s => s.trim().toLowerCase()).filter(Boolean);
}

function median(nums: number[]): number | null {
  if (!nums.length) return null;
  const s = [...nums].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 === 0 ? (s[m - 1] + s[m]) / 2 : s[m];
}

function r4(n: number): number {
  return Math.round(n * 10000) / 10000;
}

function scheduleNextScan(delayMs: number): void {
  const st = getState();
  if (st.scheduledTimeoutId !== null) {
    clearTimeout(st.scheduledTimeoutId);
    st.scheduledTimeoutId = null;
  }
  st.nextScheduledScanAt = Date.now() + delayMs;
  st.scheduledTimeoutId = setTimeout(() => {
    st.scheduledTimeoutId = null;
    void runScan().finally(() => {
      scheduleNextScan(SCAN_INTERVAL_MS());
    });
  }, delayMs);
}

/**
 * Pipeline read-only partilhada com runScan — não altera estado do probe.
 * pocket-execution-probe usa isto para dois snapshots sem duplicar regras de família/elegibilidade.
 */
async function computePocketFamilyEligibleSnapshot(): Promise<{
  allRaw: GammaRawMarket[];
  now: number;
  totalActiveInFamily: number;
  bucketMap: Map<string, PocketEconomicsMarketRow[]>;
}> {
  const allRaw = await fetchAllActive();
  const now = Date.now();
  const familyMembers: GammaRawMarket[] = [];
  for (let i = 0; i < allRaw.length; i++) {
    const raw = allRaw[i];
    if (!raw.active || raw.closed) continue;
    let outcomes: string[];
    let prices: number[];
    try {
      outcomes = JSON.parse(raw.outcomes || "[]");
      prices = JSON.parse(raw.outcomePrices || "[]").map(Number);
    } catch {
      continue;
    }
    if (outcomes.length < 2 || outcomes.length !== prices.length) continue;
    const u = classifyUnderlying(raw.question, raw.category);
    const t = classifyTemplate(raw.question, outcomes);
    const h = classifyHorizon(parseEndDate(raw.end_date_iso, raw.question), now);
    if (familyKey(u, t, h) !== TARGET_FAMILY_KEY) continue;
    familyMembers.push(raw);
    if (i > 0 && i % 1000 === 0) {
      await new Promise<void>(r => {
        setImmediate(r);
      });
    }
  }
  const totalActiveInFamily = familyMembers.length;
  const bucketMap = new Map<string, PocketEconomicsMarketRow[]>();
  for (const raw of familyMembers) {
    let outcomes: string[];
    let prices: number[];
    try {
      outcomes = JSON.parse(raw.outcomes || "[]");
      prices = JSON.parse(raw.outcomePrices || "[]").map(Number);
    } catch {
      continue;
    }
    const sorted = [...prices].sort((a, b) => b - a);
    const spread = sorted.length >= 2 ? sorted[0] - sorted[sorted.length - 1] : 0;
    const liquidity = parseFloat(raw.liquidity) || 0;
    const validPrices = prices.filter(p => p > ELIGIBLE_MIN_PRICE && p < ELIGIBLE_MAX_PRICE);
    const inPriceBand = validPrices.length >= 2;
    if (!isEligible(spread, liquidity, inPriceBand)) continue;
    const microBucketKey = normalizeQuestionStructuralKey(raw.question || "");
    const row: PocketEconomicsMarketRow = {
      id: raw.id,
      question: raw.question.slice(0, 280),
      liquidity,
      spread: r4(spread),
      prices: prices.map(p => r4(p)),
      outcomes,
    };
    const arr = bucketMap.get(microBucketKey);
    if (arr) arr.push(row);
    else bucketMap.set(microBucketKey, [row]);
  }
  return { allRaw, now, totalActiveInFamily, bucketMap };
}

/** Read-only: dois snapshots Gamma com mesma lógica que pocket-economics (sem mutar economics). */
export async function snapshotPocketFamilyEligibleBucketsForExecutionProbe(): Promise<{
  totalMarketsScanned: number;
  fetchedAt: number;
  totalActiveInFamily: number;
  bucketMap: Map<string, PocketEconomicsMarketRow[]>;
}> {
  const { allRaw, now, totalActiveInFamily, bucketMap } = await computePocketFamilyEligibleSnapshot();
  return {
    totalMarketsScanned: allRaw.length,
    fetchedAt: now,
    totalActiveInFamily,
    bucketMap,
  };
}

/** Scans consecutivos com elegibilidade >0 no histórico pocket-economics para esta micro-chave. */
export function getMicroBucketEligibleStreakFromEconomics(microBucketKey: string): number {
  const st = getState();
  let streak = 0;
  for (let i = st.eligibleHistory.length - 1; i >= 0; i--) {
    if ((st.eligibleHistory[i][microBucketKey] ?? 0) > 0) streak++;
    else break;
  }
  return streak;
}

async function runScan(): Promise<void> {
  const st = getState();
  if (st.scanning) {
    st.totalScanSkippedBusy++;
    console.warn("[PocketEconomics] runScan skipped: already scanning");
    return;
  }
  const errsBefore = st.totalScanErrors;
  let scanSucceeded = false;
  st.totalScanAttempts++;
  st.currentRunId++;
  const runId = st.currentRunId;
  st.lastScanStartAt = Date.now();
  const t0 = Date.now();
  const prioritySet = new Set(parsePriorityKeys());

  try {
    st.scanning = true;
    st.scanError = null;
    const { allRaw, now, totalActiveInFamily, bucketMap } = await computePocketFamilyEligibleSnapshot();
    st.totalMarketsScanned = allRaw.length;

    const currentEligible: Record<string, number> = {};
    for (const [k, rows] of Array.from(bucketMap.entries())) {
      currentEligible[k] = rows.length;
    }

    const prev = st.previousEligibleByBucket;
    const currKeys = new Set(Object.keys(currentEligible).filter(k => currentEligible[k] > 0));
    const prevKeys = new Set(Object.keys(prev).filter(k => prev[k] > 0));

    const repeatedMicroBuckets: string[] = [];
    const newMicroBuckets: string[] = [];
    const droppedMicroBuckets: string[] = [];
    for (const k of Array.from(currKeys)) {
      if (prevKeys.has(k)) repeatedMicroBuckets.push(k);
      else newMicroBuckets.push(k);
    }
    for (const k of Array.from(prevKeys)) {
      if (!currKeys.has(k)) droppedMicroBuckets.push(k);
    }
    repeatedMicroBuckets.sort();
    newMicroBuckets.sort();
    droppedMicroBuckets.sort();

    st.eligibleHistory.push({ ...currentEligible });
    while (st.eligibleHistory.length > HISTORY_MAX()) st.eligibleHistory.shift();

    const nStable = STABLE_SCANS();
    const hist = st.eligibleHistory;
    const stableMicroBuckets: string[] = [];
    if (hist.length >= nStable) {
      const window = hist.slice(-nStable);
      const candidateKeys = new Set<string>();
      for (const w of window) {
        Object.keys(w).forEach(k => {
          if (w[k] > 0) candidateKeys.add(k);
        });
      }
      for (const k of Array.from(candidateKeys)) {
        const ok = window.every(w => (w[k] ?? 0) > 0);
        if (ok) stableMicroBuckets.push(k);
      }
      stableMicroBuckets.sort();
    }

    let sumStable = 0;
    let largestStable = 0;
    for (const k of stableMicroBuckets) {
      const c = currentEligible[k] ?? 0;
      sumStable += c;
      if (c > largestStable) largestStable = c;
    }
    const averageEligibleCountInStablePocket =
      stableMicroBuckets.length > 0 ? r4(sumStable / stableMicroBuckets.length) : null;

    const bucketsWithEligible = currKeys.size;
    const stablePocketShare =
      bucketsWithEligible > 0 ? r4(stableMicroBuckets.length / bucketsWithEligible) : null;

    let persistSum = 0;
    let persistN = 0;
    const maxStreakCap = Math.max(1, hist.length);
    for (const k of Array.from(currKeys)) {
      let streak = 0;
      for (let i = hist.length - 1; i >= 0; i--) {
        if ((hist[i][k] ?? 0) > 0) streak++;
        else break;
      }
      persistSum += streak;
      persistN++;
    }
    const pocketPersistenceScore =
      persistN > 0 ? r4(persistSum / (persistN * maxStreakCap)) : null;

    const buckets: PocketEconomicsBucketRow[] = [];
    for (const [microBucketKey, markets] of Array.from(bucketMap.entries())) {
      const spreads = markets.map(m => m.spread);
      buckets.push({
        microBucketKey,
        eligibleCount: markets.length,
        prioritized: prioritySet.has(microBucketKey),
        marketIds: markets.map(m => m.id),
        markets,
        spreadProxyMedian: median(spreads),
        localDensity: totalActiveInFamily > 0 ? r4(markets.length / totalActiveInFamily) : null,
      });
    }
    buckets.sort((a, b) => {
      if (a.prioritized !== b.prioritized) return a.prioritized ? -1 : 1;
      if (b.eligibleCount !== a.eligibleCount) return b.eligibleCount - a.eligibleCount;
      return a.microBucketKey.localeCompare(b.microBucketKey);
    });

    const plausShare = stablePocketShare ?? 0;
    const plausSize = Math.min(1, largestStable / 10);
    const plausAvg = averageEligibleCountInStablePocket !== null ? Math.min(1, averageEligibleCountInStablePocket / 5) : 0;
    const plausPersist = pocketPersistenceScore ?? 0;
    const recurrentSubstratePlausibilityScore = r4((plausShare + plausSize + plausAvg + plausPersist) / 4);

    st.previousEligibleByBucket = { ...currentEligible };
    st.previousScanTs = Date.now();
    st.lastDigestBuckets = buckets;
    const tDone = Date.now();
    st.lastScanTimestamp = tDone;
    st.lastScanDurationMs = tDone - t0;
    st.lastSuccessfulScanAt = tDone;
    st.lastScanErrorMessage = null;
    st.lastScanErrorAt = null;
    st.totalScanSuccess++;

    st.lastMetricsSnapshot = {
      repeatedMicroBuckets,
      newMicroBuckets,
      droppedMicroBuckets,
      stableMicroBuckets,
      averageEligibleCountInStablePocket,
      largestStablePocketSize: largestStable,
      stablePocketShare,
      pocketPersistenceScore,
      recurrentSubstratePlausibilityScore,
      totalActiveInFamily,
      totalEligibleInFamily: Array.from(currKeys).reduce((s, k) => s + currentEligible[k], 0),
    };

    console.log(
      `[PocketEconomics] Scan #${runId} ok: familyActive=${totalActiveInFamily} eligibleBuckets=${bucketsWithEligible} stable=${stableMicroBuckets.length}`,
    );
    scanSucceeded = true;
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    st.scanError = msg;
    st.lastScanErrorMessage = msg;
    st.lastScanErrorAt = Date.now();
    st.totalScanErrors++;
    console.error(`[PocketEconomics] Scan #${runId} error:`, msg);
  } finally {
    st.lastScanEndAt = Date.now();
    st.scanning = false;
    maybePersistPocketEconomicsState(st, scanSucceeded || st.totalScanErrors > errsBefore);
  }
}

function getLastMetrics(st: ProbeState): LastMetrics | null {
  return st.lastMetricsSnapshot ?? null;
}

function buildPromotionAssessment(
  st: ProbeState,
  m: LastMetrics | null,
): PocketEconomicsPromotionAssessment {
  const thresholdsUsed: PocketEconomicsPromotionThresholdsDigest = {
    minEligibleHistoryScans: PROMO_MIN_ELIGIBLE_HISTORY_SCANS(),
    minStablePocketShare: PROMO_MIN_STABLE_POCKET_SHARE(),
    minRepeatedMicroBuckets: PROMO_MIN_REPEATED_MICRO_BUCKETS(),
    minAverageEligibleStable: PROMO_MIN_AVG_ELIGIBLE_STABLE(),
    minLargestStablePocketSize: PROMO_MIN_LARGEST_STABLE_SIZE(),
    minPersistenceScore: PROMO_MIN_PERSISTENCE_SCORE(),
  };

  const histLen = st.eligibleHistory.length;
  const minHist = thresholdsUsed.minEligibleHistoryScans;
  const minimumStableScansSatisfied = histLen >= minHist;

  const share = m?.stablePocketShare ?? null;
  const stablePocketShareSatisfied =
    share !== null && share >= thresholdsUsed.minStablePocketShare;

  const repeated = m?.repeatedMicroBuckets.length ?? 0;
  const minimumRepeatedMicroBucketsSatisfied =
    repeated >= thresholdsUsed.minRepeatedMicroBuckets;

  const avg = m?.averageEligibleCountInStablePocket ?? null;
  const minimumAverageEligibleCountSatisfied =
    avg !== null && avg >= thresholdsUsed.minAverageEligibleStable;

  const largest = m?.largestStablePocketSize ?? 0;
  const largestStablePocketSizeSatisfied =
    largest >= thresholdsUsed.minLargestStablePocketSize;

  const pScore = m?.pocketPersistenceScore ?? null;
  const persistenceScoreSatisfied =
    pScore !== null && pScore >= thresholdsUsed.minPersistenceScore;

  const satisfiedCount = [
    minimumStableScansSatisfied,
    stablePocketShareSatisfied,
    minimumRepeatedMicroBucketsSatisfied,
    minimumAverageEligibleCountSatisfied,
    largestStablePocketSizeSatisfied,
    persistenceScoreSatisfied,
  ].filter(Boolean).length;

  const stableBucketCount = m?.stableMicroBuckets.length ?? 0;
  const hasStableSubstrate = stableBucketCount > 0;

  const promotionReasons: string[] = [];
  const blockingReasons: string[] = [];

  if (minimumStableScansSatisfied) {
    promotionReasons.push(
      `eligibleHistory depth ${histLen} >= ${minHist} (stable-window evidence)`,
    );
  } else {
    blockingReasons.push(
      `eligibleHistory depth ${histLen} < ${minHist} (need more scans for stable-window metrics)`,
    );
  }

  if (stablePocketShareSatisfied) {
    promotionReasons.push(
      `stablePocketShare ${r4(share!)} >= ${thresholdsUsed.minStablePocketShare}`,
    );
  } else {
    blockingReasons.push(
      share === null
        ? "stablePocketShare null (no eligible-bucket share computed)"
        : `stablePocketShare ${r4(share)} < ${thresholdsUsed.minStablePocketShare}`,
    );
  }

  if (minimumRepeatedMicroBucketsSatisfied) {
    promotionReasons.push(
      `repeatedMicroBuckets count ${repeated} >= ${thresholdsUsed.minRepeatedMicroBuckets}`,
    );
  } else {
    blockingReasons.push(
      `repeatedMicroBuckets count ${repeated} < ${thresholdsUsed.minRepeatedMicroBuckets}`,
    );
  }

  if (minimumAverageEligibleCountSatisfied) {
    promotionReasons.push(
      `averageEligibleCountInStablePocket ${r4(avg!)} >= ${thresholdsUsed.minAverageEligibleStable}`,
    );
  } else {
    blockingReasons.push(
      avg === null
        ? "averageEligibleCountInStablePocket null (no stable micro-buckets)"
        : `averageEligibleCountInStablePocket ${r4(avg)} < ${thresholdsUsed.minAverageEligibleStable}`,
    );
  }

  if (largestStablePocketSizeSatisfied) {
    promotionReasons.push(
      `largestStablePocketSize ${largest} >= ${thresholdsUsed.minLargestStablePocketSize}`,
    );
  } else {
    blockingReasons.push(
      `largestStablePocketSize ${largest} < ${thresholdsUsed.minLargestStablePocketSize}`,
    );
  }

  if (persistenceScoreSatisfied) {
    promotionReasons.push(
      `pocketPersistenceScore ${r4(pScore!)} >= ${thresholdsUsed.minPersistenceScore}`,
    );
  } else {
    blockingReasons.push(
      pScore === null
        ? "pocketPersistenceScore null"
        : `pocketPersistenceScore ${r4(pScore)} < ${thresholdsUsed.minPersistenceScore}`,
    );
  }

  let overallPromotionVerdict: PocketEconomicsPromotionVerdict;
  if (!minimumStableScansSatisfied || !hasStableSubstrate) {
    overallPromotionVerdict = "not_ready";
  } else if (satisfiedCount === 6) {
    overallPromotionVerdict = "ready_for_minimal_execution_probe";
  } else if (satisfiedCount >= 4) {
    overallPromotionVerdict = "borderline";
  } else {
    overallPromotionVerdict = "not_ready";
  }

  return {
    minimumStableScansSatisfied,
    stablePocketShareSatisfied,
    minimumRepeatedMicroBucketsSatisfied,
    minimumAverageEligibleCountSatisfied,
    largestStablePocketSizeSatisfied,
    persistenceScoreSatisfied,
    overallPromotionVerdict,
    promotionReasons,
    blockingReasons,
    thresholdsUsed,
  };
}

export function ensurePocketEconomicsProbe(): void {
  const st = getState();
  if (st.loopStarted) return;
  st.loopStarted = true;
  st.schedulerStartedAt = Date.now();
  console.log("[PocketEconomics] Scheduler started (setTimeout chain)");
  scheduleNextScan(3000);
}

export function getPocketEconomicsHealth(): {
  pocketEconomicsSchedulerRunning: boolean;
  pocketEconomicsIsScanRunning: boolean;
  lastSuccessfulPocketEconomicsScanAt: string | null;
  lastPocketEconomicsErrorAt: string | null;
  pocketEconomicsSchedulerStartedAt: string | null;
  pocketEconomicsLastScanStartAt: string | null;
  pocketEconomicsLastScanEndAt: string | null;
  pocketEconomicsLastScanErrorMessage: string | null;
  pocketEconomicsNextScheduledScanAt: string | null;
  pocketEconomicsCurrentRunId: number;
  pocketEconomicsTotalScanAttempts: number;
  pocketEconomicsTotalScanSuccess: number;
  pocketEconomicsTotalScanErrors: number;
  pocketEconomicsTotalScanSkippedBusy: number;
  pocketEconomicsPersistenceEnabled: boolean;
  pocketEconomicsPersistencePath: string | null;
  pocketEconomicsLastPersistenceWriteAt: string | null;
  pocketEconomicsLastPersistenceLoadAt: string | null;
  pocketEconomicsPersistenceLoadStatus: PocketEconomicsPersistenceLoadStatus;
} {
  const st = getState();
  const p = buildPersistenceDigest();
  return {
    pocketEconomicsSchedulerRunning: st.loopStarted,
    pocketEconomicsIsScanRunning: st.scanning,
    lastSuccessfulPocketEconomicsScanAt: st.lastSuccessfulScanAt ? new Date(st.lastSuccessfulScanAt).toISOString() : null,
    lastPocketEconomicsErrorAt: st.lastScanErrorAt ? new Date(st.lastScanErrorAt).toISOString() : null,
    pocketEconomicsSchedulerStartedAt: st.schedulerStartedAt ? new Date(st.schedulerStartedAt).toISOString() : null,
    pocketEconomicsLastScanStartAt: st.lastScanStartAt ? new Date(st.lastScanStartAt).toISOString() : null,
    pocketEconomicsLastScanEndAt: st.lastScanEndAt ? new Date(st.lastScanEndAt).toISOString() : null,
    pocketEconomicsLastScanErrorMessage: st.lastScanErrorMessage,
    pocketEconomicsNextScheduledScanAt: st.nextScheduledScanAt ? new Date(st.nextScheduledScanAt).toISOString() : null,
    pocketEconomicsCurrentRunId: st.currentRunId,
    pocketEconomicsTotalScanAttempts: st.totalScanAttempts,
    pocketEconomicsTotalScanSuccess: st.totalScanSuccess,
    pocketEconomicsTotalScanErrors: st.totalScanErrors,
    pocketEconomicsTotalScanSkippedBusy: st.totalScanSkippedBusy,
    pocketEconomicsPersistenceEnabled: p.persistenceEnabled,
    pocketEconomicsPersistencePath: p.persistencePath,
    pocketEconomicsLastPersistenceWriteAt: p.lastPersistenceWriteAt,
    pocketEconomicsLastPersistenceLoadAt: p.lastPersistenceLoadAt,
    pocketEconomicsPersistenceLoadStatus: p.persistenceLoadStatus,
  };
}

export function buildPocketEconomicsDigest(): PocketEconomicsDigest {
  const st = getState();
  const scanStatus: PocketEconomicsDigest["scanStatus"] =
    st.scanError ? "error" : st.scanning ? "scanning" : st.lastScanTimestamp ? "completed" : "idle";

  const m = getLastMetrics(st);
  const emptyBuckets: PocketEconomicsBucketRow[] = [];

  return {
    computedAt: new Date().toISOString(),
    probeVersion: "pocket-economics-v1",
    scanStatus,
    targetFamilyKey: TARGET_FAMILY_KEY,
    note:
      "Observacional apenas. Família fixa other:price_above:>3m; mesma elegibilidade que catalog-pocket (spread<0.15, liq>=500, 2 outcomes em (0.03,0.97)). recurrentSubstratePlausibilityScore é heurística 0–1 — não é edge nem PnL. promotionAssessment: gate explícito (não executa trades) para probe de execução mínimo; thresholds POCKET_ECON_PROMO_*. POCKET_ECON_PRIORITY_MICRO_BUCKETS: lista opcional. Persistência: pocket-economics-state.json sob PAPER_STATE_DIR ou cwd/.paper (POCKET_ECON_STATE_PATH, POCKET_ECON_DISABLE_DISK=1).",
    prioritizedMicroBucketKeys: parsePriorityKeys(),
    repeatedMicroBuckets: m?.repeatedMicroBuckets ?? [],
    newMicroBuckets: m?.newMicroBuckets ?? [],
    droppedMicroBuckets: m?.droppedMicroBuckets ?? [],
    stableMicroBuckets: m?.stableMicroBuckets ?? [],
    averageEligibleCountInStablePocket: m?.averageEligibleCountInStablePocket ?? null,
    largestStablePocketSize: m?.largestStablePocketSize ?? 0,
    stablePocketShare: m?.stablePocketShare ?? null,
    pocketPersistenceScore: m?.pocketPersistenceScore ?? null,
    recurrentSubstratePlausibilityScore: m?.recurrentSubstratePlausibilityScore ?? null,
    currentCycle: {
      totalActiveInFamily: m?.totalActiveInFamily ?? 0,
      totalEligibleInFamily: m?.totalEligibleInFamily ?? 0,
      buckets: st.lastDigestBuckets.length ? st.lastDigestBuckets : emptyBuckets,
    },
    previousScanTimestamp: st.previousScanTs ? new Date(st.previousScanTs).toISOString() : null,
    scanDurationMs: st.lastScanDurationMs,
    scanTimestamp: st.lastScanTimestamp ? new Date(st.lastScanTimestamp).toISOString() : null,
    totalMarketsScanned: st.totalMarketsScanned,
    lastScanStartAt: st.lastScanStartAt ? new Date(st.lastScanStartAt).toISOString() : null,
    lastScanEndAt: st.lastScanEndAt ? new Date(st.lastScanEndAt).toISOString() : null,
    lastSuccessfulScanAt: st.lastSuccessfulScanAt ? new Date(st.lastSuccessfulScanAt).toISOString() : null,
    lastScanErrorAt: st.lastScanErrorAt ? new Date(st.lastScanErrorAt).toISOString() : null,
    lastScanErrorMessage: st.lastScanErrorMessage,
    isScanRunning: st.scanning,
    currentRunId: st.currentRunId,
    nextScheduledScanAt: st.nextScheduledScanAt ? new Date(st.nextScheduledScanAt).toISOString() : null,
    schedulerStartedAt: st.schedulerStartedAt ? new Date(st.schedulerStartedAt).toISOString() : null,
    totalScanAttempts: st.totalScanAttempts,
    totalScanSuccess: st.totalScanSuccess,
    totalScanErrors: st.totalScanErrors,
    totalScanSkippedBusy: st.totalScanSkippedBusy,
    persistence: buildPersistenceDigest(),
    promotionAssessment: buildPromotionAssessment(st, m),
  };
}
