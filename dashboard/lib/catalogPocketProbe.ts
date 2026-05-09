/**
 * Catalog Pocket Probe — local recurrent pockets within broad families
 *
 * Mesma base que catalog_reality_probe: (underlying, template, horizon_bucket).
 * Micro-bucket V1.1: chave estrutural normalizada da `question` (ver normalizeQuestionStructuralKey).
 *
 * NÃO mede edge, latência, PnL. NÃO trades/paper.
 */

function envNum(k: string, d: number): number {
  const v = Number(process.env[k]?.trim());
  return Number.isFinite(v) ? v : d;
}

const GAMMA_URL = "https://gamma-api.polymarket.com/markets";
const GAMMA_FETCH_TIMEOUT_MS = 15_000;
const SCAN_INTERVAL_MS = () => envNum("CATALOG_PROBE_SCAN_INTERVAL_MS", 6 * 3600_000);

const ELIGIBLE_MAX_SPREAD = 0.15;
const ELIGIBLE_MIN_LIQ = 500;
const ELIGIBLE_MIN_PRICE = 0.03;
const ELIGIBLE_MAX_PRICE = 0.97;

const N_MIN = () => envNum("CATALOG_POCKET_N_MIN", 3);
const R_MIN = () => envNum("CATALOG_POCKET_R_MIN", 2);
const W_SCANS = () => envNum("CATALOG_POCKET_W_SCANS", 3);
const K_BUCKETS = () => envNum("CATALOG_POCKET_K_BUCKETS", 50);
/** Truncagem final do microBucketKey para estabilidade do digest (determinística). */
const MICRO_BUCKET_KEY_MAX_LEN = () => envNum("CATALOG_POCKET_KEY_MAX_LEN", 192);

type Underlying = "BTC" | "ETH" | "SOL" | "political" | "sports" | "other_crypto" | "other";
type Template = "price_above" | "price_below" | "price_range" | "yes_no_event" | "multi_outcome" | "other";
type HorizonBucket = "<24h" | "24h-168h" | "1w-4w" | "1m-3m" | ">3m" | "unknown";

export type FamilyPocketVerdict = "no_pocket" | "ephemeral_pocket" | "recurrent_pocket" | "pocket_degraded";
export type GlobalPocketVerdict = "no_actionable_substrate" | "localized_substrate_exists";

interface GammaRawMarket {
  id: string; question: string; outcomes: string; outcomePrices: string;
  liquidity: string; volume: string; active: boolean; closed: boolean;
  market_slug: string; end_date_iso?: string; category?: string;
}

interface ParsedMarket {
  underlying: Underlying;
  template: Template;
  horizonBucket: HorizonBucket;
  liquidity: number;
  spread: number;
  inPriceBand: boolean;
  microBucketKey: string;
}

export interface FamilyPocketRow {
  familyKey: string;
  underlying: Underlying;
  template: Template;
  horizonBucket: HorizonBucket;
  activeMarketCount: number;
  eligibleCountInFamily: number;
  largestPocketSize: number;
  pocketCount_ge_Nmin: number;
  localDensity: number | null;
  recurrenceScore: number;
  pocketStabilityKey: string[];
  familyPocketVerdict: FamilyPocketVerdict;
}

export interface CatalogPocketDigest {
  computedAt: string;
  probeVersion: "catalog-pocket-v1.1";
  scanStatus: "completed" | "scanning" | "idle" | "error";
  globalVerdict: GlobalPocketVerdict;
  familiesWithRecurringPocket: number;
  scanDurationMs: number | null;
  scanTimestamp: string | null;
  totalMarketsScanned: number;
  totalActiveNonClosed: number;
  config: {
    nMin: number;
    rMin: number;
    wScans: number;
    kBuckets: number;
    microBucketKeyMaxLen: number;
    scanIntervalMs: number;
  };
  familyRows: FamilyPocketRow[];
  topPockets: Array<{ familyKey: string; microBucketKey: string; eligibleCount: number }>;
  note: string;
  /** Observabilidade scheduler/runtime — não altera critérios económicos */
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
}

const GLOBAL_KEY = "__catalogPocketProbe_v1";

interface ScanSnapshot {
  ts: number;
  /** familyKey -> largestPocketSize neste scan */
  largestByFamily: Record<string, number>;
  /** familyKey -> Set de microBucket keys com count >= N_min */
  pocketKeysByFamily: Record<string, Set<string>>;
}

interface ProbeState {
  loopStarted: boolean;
  scanning: boolean;
  scanError: string | null;
  lastScanTimestamp: number | null;
  lastScanDurationMs: number | null;
  totalMarketsScanned: number;
  totalActiveNonClosed: number;
  /** últimos W scans (mais recente no fim) */
  scanHistory: ScanSnapshot[];
  /** familyKey -> último familyPocketVerdict emitido */
  lastFamilyVerdict: Record<string, FamilyPocketVerdict>;
  familyRows: FamilyPocketRow[];
  topPockets: Array<{ familyKey: string; microBucketKey: string; eligibleCount: number }>;
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
}

function getState(): ProbeState {
  const g = globalThis as unknown as Record<string, ProbeState | undefined>;
  if (!g[GLOBAL_KEY]) {
    g[GLOBAL_KEY] = {
      loopStarted: false, scanning: false, scanError: null,
      lastScanTimestamp: null, lastScanDurationMs: null,
      totalMarketsScanned: 0, totalActiveNonClosed: 0,
      scanHistory: [], lastFamilyVerdict: {},
      familyRows: [], topPockets: [],
      lastScanStartAt: null, lastScanEndAt: null, lastSuccessfulScanAt: null,
      lastScanErrorAt: null, lastScanErrorMessage: null,
      currentRunId: 0, nextScheduledScanAt: null, schedulerStartedAt: null,
      scheduledTimeoutId: null,
      totalScanAttempts: 0, totalScanSuccess: 0, totalScanErrors: 0, totalScanSkippedBusy: 0,
    };
  }
  const st = g[GLOBAL_KEY]!;
  if (!st.scanHistory) st.scanHistory = [];
  if (!st.lastFamilyVerdict) st.lastFamilyVerdict = {};
  if (st.lastScanStartAt === undefined) st.lastScanStartAt = null;
  if (st.lastScanEndAt === undefined) st.lastScanEndAt = null;
  if (st.lastSuccessfulScanAt === undefined) st.lastSuccessfulScanAt = null;
  if (st.lastScanErrorAt === undefined) st.lastScanErrorAt = null;
  if (st.lastScanErrorMessage === undefined) st.lastScanErrorMessage = null;
  if (st.currentRunId === undefined) st.currentRunId = 0;
  if (st.nextScheduledScanAt === undefined) st.nextScheduledScanAt = null;
  if (st.schedulerStartedAt === undefined) st.schedulerStartedAt = null;
  if (st.scheduledTimeoutId === undefined) st.scheduledTimeoutId = null;
  if (st.totalScanAttempts === undefined) st.totalScanAttempts = 0;
  if (st.totalScanSuccess === undefined) st.totalScanSuccess = 0;
  if (st.totalScanErrors === undefined) st.totalScanErrors = 0;
  if (st.totalScanSkippedBusy === undefined) st.totalScanSkippedBusy = 0;
  return st;
}

// --- mesma lógica que catalog_reality_probe (classificação + elegibilidade) ---

const DATE_IN_QUESTION_RE = /\b(?:by|on|before)\s+(?:(\w+ \d{1,2},?\s*\d{4})|(\w+ \d{4})|(\d{4}-\d{2}-\d{2}))\b/i;

/**
 * Valores monetários parseáveis: $ opcionalmente seguido de dígitos com vírgulas/decimais e sufixo k/m/M.
 * Ordem: substituição sequencial por $_STRIKE1, $_STRIKE2, … (não colapsar num único placeholder).
 */
const PARSEABLE_MONETARY_RE = /\$\s?[\d,]+(?:\.\d+)?\s*[kKmM]?/gi;

/**
 * V1.1 micro-bucket: texto da pergunta normalizado estruturalmente.
 * - lowercase, espaços normalizados, trim
 * - aspas “smart” → ASCII; remoção de zero-width
 * - valores $ substituídos em ordem por $_STRIKEn
 * - datas não normalizadas (mantêm-se como no texto após lowercase)
 * - truncagem final determinística
 */
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

function buildMicroBucketKey(raw: GammaRawMarket): string {
  return normalizeQuestionStructuralKey(raw.question || "");
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
    const raw = m[1] ?? m[2] ?? m[3] ?? "";
    const d = new Date(raw);
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

function parseMarket(raw: GammaRawMarket, now: number): ParsedMarket | null {
  if (!raw.active || raw.closed) return null;
  let outcomes: string[];
  let prices: number[];
  try {
    outcomes = JSON.parse(raw.outcomes || "[]");
    prices = JSON.parse(raw.outcomePrices || "[]").map(Number);
  } catch { return null; }
  if (outcomes.length < 2 || outcomes.length !== prices.length) return null;

  const liquidity = parseFloat(raw.liquidity) || 0;
  const sorted = [...prices].sort((a, b) => b - a);
  const spread = sorted.length >= 2 ? sorted[0] - sorted[sorted.length - 1] : 0;
  const validPrices = prices.filter(p => p > ELIGIBLE_MIN_PRICE && p < ELIGIBLE_MAX_PRICE);
  const inPriceBand = validPrices.length >= 2;

  const underlying = classifyUnderlying(raw.question, raw.category);
  const template = classifyTemplate(raw.question, outcomes);
  const endDate = parseEndDate(raw.end_date_iso, raw.question);
  const horizonBucket = classifyHorizon(endDate, now);
  const microBucketKey = buildMicroBucketKey(raw);

  return {
    underlying, template, horizonBucket, liquidity, spread, inPriceBand,
    microBucketKey,
  };
}

function isEligible(m: ParsedMarket): boolean {
  return m.spread < ELIGIBLE_MAX_SPREAD && m.liquidity >= ELIGIBLE_MIN_LIQ && m.inPriceBand;
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
      await new Promise<void>(resolve => {
        setImmediate(resolve);
      });
    }
  }
  return all;
}

function groupByFamily(markets: ParsedMarket[]): Map<string, ParsedMarket[]> {
  const g = new Map<string, ParsedMarket[]>();
  for (const m of markets) {
    const key = `${m.underlying}:${m.template}:${m.horizonBucket}`;
    const arr = g.get(key);
    if (arr) arr.push(m); else g.set(key, [m]);
  }
  return g;
}

function analyzeFamily(
  familyKey: string,
  members: ParsedMarket[],
  eligible: ParsedMarket[],
  bucketCounts: Map<string, number>,
  nMin: number,
  kMax: number,
): {
  activeMarketCount: number;
  eligibleCountInFamily: number;
  largestPocketSize: number;
  pocketCount_ge_Nmin: number;
  localDensity: number | null;
  topBuckets: Array<{ microBucketKey: string; eligibleCount: number }>;
} {
  const activeMarketCount = members.length;
  const eligibleCountInFamily = eligible.length;
  const sortedBuckets = Array.from(bucketCounts.entries())
    .sort((a, b) => b[1] - a[1]);
  const largestPocketSize = sortedBuckets.length ? sortedBuckets[0][1] : 0;
  const pocketCount_ge_Nmin = sortedBuckets.filter(([, c]) => c >= nMin).length;
  const localDensity = activeMarketCount > 0 ? Math.round((largestPocketSize / activeMarketCount) * 10000) / 10000 : null;
  const topBuckets = sortedBuckets.slice(0, kMax).map(([microBucketKey, eligibleCount]) => ({ microBucketKey, eligibleCount }));
  return {
    activeMarketCount, eligibleCountInFamily, largestPocketSize, pocketCount_ge_Nmin,
    localDensity, topBuckets,
  };
}

/** Janela de W scans: últimos (W-1) no histórico + scan actual. */
function scanWindow(history: ScanSnapshot[], currentSnap: ScanSnapshot, w: number): ScanSnapshot[] {
  return [...history.slice(-(w - 1)), currentSnap];
}

function computeRecurrenceScore(
  window: ScanSnapshot[],
  familyKey: string,
  nMin: number,
): number {
  let score = 0;
  for (const snap of window) {
    const v = snap.largestByFamily[familyKey];
    if (v !== undefined && v >= nMin) score++;
  }
  return score;
}

function computePocketStabilityKey(
  window: ScanSnapshot[],
  familyKey: string,
  rMin: number,
): string[] {
  const keyCounts = new Map<string, number>();
  for (const snap of window) {
    const set = snap.pocketKeysByFamily[familyKey];
    if (!set) continue;
    set.forEach(k => {
      keyCounts.set(k, (keyCounts.get(k) ?? 0) + 1);
    });
  }
  return Array.from(keyCounts.entries()).filter(([, c]) => c >= rMin).map(([k]) => k).sort();
}

function decideVerdict(
  largestPocketSize: number,
  recurrenceScore: number,
  prev: FamilyPocketVerdict | undefined,
  nMin: number,
  rMin: number,
): FamilyPocketVerdict {
  const hasPocketNow = largestPocketSize >= nMin;
  const recurrent = recurrenceScore >= rMin;

  if (hasPocketNow && recurrent) return "recurrent_pocket";
  if (hasPocketNow && !recurrent) return "ephemeral_pocket";
  if (!hasPocketNow && recurrent) return "pocket_degraded";
  if (!hasPocketNow && prev === "recurrent_pocket") return "pocket_degraded";
  return "no_pocket";
}

/**
 * Agenda o próximo scan para `delayMs` a partir de agora.
 * Cadeia setTimeout (não setInterval): o próximo ciclo só começa depois do anterior terminar — evita ticks sobrepostos silenciosos.
 */
function scheduleNextCatalogPocketScan(delayMs: number): void {
  const st = getState();
  if (st.scheduledTimeoutId !== null) {
    clearTimeout(st.scheduledTimeoutId);
    st.scheduledTimeoutId = null;
  }
  st.nextScheduledScanAt = Date.now() + delayMs;
  st.scheduledTimeoutId = setTimeout(() => {
    st.scheduledTimeoutId = null;
    void runScan().finally(() => {
      scheduleNextCatalogPocketScan(SCAN_INTERVAL_MS());
    });
  }, delayMs);
}

async function runScan(): Promise<void> {
  const st = getState();
  if (st.scanning) {
    st.totalScanSkippedBusy++;
    console.warn("[CatalogPocket] runScan skipped: scan already in progress");
    return;
  }
  st.totalScanAttempts++;
  st.currentRunId++;
  const runId = st.currentRunId;
  const tStart = Date.now();
  st.lastScanStartAt = tStart;
  const t0 = Date.now();
  const nMin = N_MIN();
  const w = W_SCANS();
  const kMax = K_BUCKETS();

  try {
    st.scanning = true;
    st.scanError = null;
    const allRaw = await fetchAllActive();
    st.totalMarketsScanned = allRaw.length;
    const now = Date.now();
    const parsed: ParsedMarket[] = [];
    for (let i = 0; i < allRaw.length; i++) {
      const m = parseMarket(allRaw[i], now);
      if (m) parsed.push(m);
      if (i > 0 && i % 1000 === 0) {
        await new Promise<void>(resolve => {
          setImmediate(resolve);
        });
      }
    }
    st.totalActiveNonClosed = parsed.length;

    const byFam = groupByFamily(parsed);
    const largestByFamily: Record<string, number> = {};
    const pocketKeysByFamily: Record<string, Set<string>> = {};
    const allTopPockets: Array<{ familyKey: string; microBucketKey: string; eligibleCount: number }> = [];

    const familyAnalysis = new Map<string, ReturnType<typeof analyzeFamily> & { underlying: Underlying; template: Template; horizonBucket: HorizonBucket }>();

    for (const [familyKey, members] of Array.from(byFam.entries())) {
      const eligible = members.filter(isEligible);
      const bucketCounts = new Map<string, number>();
      for (const e of eligible) {
        bucketCounts.set(e.microBucketKey, (bucketCounts.get(e.microBucketKey) ?? 0) + 1);
      }

      const [underlying, template, horizonBucket] = familyKey.split(":") as [Underlying, Template, HorizonBucket];
      const a = analyzeFamily(familyKey, members, eligible, bucketCounts, nMin, kMax);

      largestByFamily[familyKey] = a.largestPocketSize;
      const keysWithPocket = new Set<string>();
      for (const [bk, c] of Array.from(bucketCounts.entries())) {
        if (c >= nMin) keysWithPocket.add(bk);
      }
      pocketKeysByFamily[familyKey] = keysWithPocket;

      for (const tb of a.topBuckets) {
        allTopPockets.push({ familyKey, microBucketKey: tb.microBucketKey, eligibleCount: tb.eligibleCount });
      }

      familyAnalysis.set(familyKey, { ...a, underlying, template, horizonBucket });
    }

    const snap: ScanSnapshot = {
      ts: Date.now(),
      largestByFamily,
      pocketKeysByFamily,
    };
    const window = scanWindow(st.scanHistory, snap, w);
    const rMinVal = R_MIN();

    const rows: FamilyPocketRow[] = [];
    for (const [familyKey, a] of Array.from(familyAnalysis.entries())) {
      const recurrenceScore = computeRecurrenceScore(window, familyKey, nMin);
      const pocketStabilityKey = computePocketStabilityKey(window, familyKey, rMinVal);

      const prev = st.lastFamilyVerdict[familyKey];
      const verdict = decideVerdict(a.largestPocketSize, recurrenceScore, prev, nMin, rMinVal);
      st.lastFamilyVerdict[familyKey] = verdict;

      rows.push({
        familyKey,
        underlying: a.underlying,
        template: a.template,
        horizonBucket: a.horizonBucket,
        activeMarketCount: a.activeMarketCount,
        eligibleCountInFamily: a.eligibleCountInFamily,
        largestPocketSize: a.largestPocketSize,
        pocketCount_ge_Nmin: a.pocketCount_ge_Nmin,
        localDensity: a.localDensity,
        recurrenceScore,
        pocketStabilityKey,
        familyPocketVerdict: verdict,
      });
    }

    rows.sort((a, b) => b.largestPocketSize - a.largestPocketSize || b.eligibleCountInFamily - a.eligibleCountInFamily);

    st.scanHistory.push(snap);
    while (st.scanHistory.length > w) st.scanHistory.shift();

    st.familyRows = rows;
    allTopPockets.sort((a, b) => b.eligibleCount - a.eligibleCount);
    st.topPockets = allTopPockets.slice(0, 50);

    const tDone = Date.now();
    st.lastScanTimestamp = tDone;
    st.lastScanDurationMs = tDone - t0;
    st.lastSuccessfulScanAt = tDone;
    st.lastScanErrorMessage = null;
    st.lastScanErrorAt = null;
    st.totalScanSuccess++;

    const recurrentFamilies = rows.filter(r => r.familyPocketVerdict === "recurrent_pocket").length;
    console.log(`[CatalogPocket] Scan #${runId} ok: ${allRaw.length} raw, ${rows.length} families, recurrent_pocket families=${recurrentFamilies}`);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    st.scanError = msg;
    st.lastScanErrorMessage = msg;
    st.lastScanErrorAt = Date.now();
    st.totalScanErrors++;
    console.error(`[CatalogPocket] Scan #${runId} error:`, msg);
  } finally {
    st.lastScanEndAt = Date.now();
    st.scanning = false;
  }
}

export function ensureCatalogPocketProbe(): void {
  const st = getState();
  if (st.loopStarted) return;
  st.loopStarted = true;
  st.schedulerStartedAt = Date.now();
  console.log("[CatalogPocket] Scheduler started (setTimeout chain after each scan; boot via instrumentation or GET /api/probe/catalog-pocket)");
  scheduleNextCatalogPocketScan(2500);
}

/** Para /api/healthz — leve, só lê estado */
export function getCatalogPocketHealth(): {
  catalogPocketSchedulerRunning: boolean;
  catalogPocketIsScanRunning: boolean;
  lastSuccessfulCatalogPocketScanAt: string | null;
  lastCatalogPocketErrorAt: string | null;
} {
  const st = getState();
  return {
    catalogPocketSchedulerRunning: st.loopStarted,
    catalogPocketIsScanRunning: st.scanning,
    lastSuccessfulCatalogPocketScanAt: st.lastSuccessfulScanAt ? new Date(st.lastSuccessfulScanAt).toISOString() : null,
    lastCatalogPocketErrorAt: st.lastScanErrorAt ? new Date(st.lastScanErrorAt).toISOString() : null,
  };
}

export function buildCatalogPocketDigest(): CatalogPocketDigest {
  const st = getState();
  const scanStatus: CatalogPocketDigest["scanStatus"] =
    st.scanError ? "error" : st.scanning ? "scanning" : st.lastScanTimestamp ? "completed" : "idle";

  const recurrent = st.familyRows.filter(r => r.familyPocketVerdict === "recurrent_pocket").length;
  const globalVerdict: GlobalPocketVerdict =
    recurrent > 0 ? "localized_substrate_exists" : "no_actionable_substrate";

  return {
    computedAt: new Date().toISOString(),
    probeVersion: "catalog-pocket-v1.1",
    scanStatus,
    globalVerdict,
    familiesWithRecurringPocket: recurrent,
    scanDurationMs: st.lastScanDurationMs,
    scanTimestamp: st.lastScanTimestamp ? new Date(st.lastScanTimestamp).toISOString() : null,
    totalMarketsScanned: st.totalMarketsScanned,
    totalActiveNonClosed: st.totalActiveNonClosed,
    config: {
      nMin: N_MIN(),
      rMin: R_MIN(),
      wScans: W_SCANS(),
      kBuckets: K_BUCKETS(),
      microBucketKeyMaxLen: MICRO_BUCKET_KEY_MAX_LEN(),
      scanIntervalMs: SCAN_INTERVAL_MS(),
    },
    familyRows: st.familyRows,
    topPockets: st.topPockets,
    note:
      "catalog-pocket-v1.1. Micro-bucket = normalizeQuestionStructuralKey(question): lowercase, collapse spaces, trim, zero-width + smart-quote cleanup, ALL parseable $ amounts replaced in order by $_STRIKE1, $_STRIKE2, …; dates unchanged; key truncated to microBucketKeyMaxLen. Same eligibility as catalog_reality. recurrenceScore / recurrent_pocket unchanged. Scheduler: setTimeout chain from process boot (instrumentation) + next scan after previous completes; yields during Gamma pagination.",
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
  };
}
