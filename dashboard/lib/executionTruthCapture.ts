/**
 * Execution Truth Capture — medições empíricas robot-native via snapshots Gamma
 * (fetch por id repetido). Sem trading live, sem workflow humano.
 * Serve para comparar custos de execução observados localmente vs proxies estáticos.
 */

import { fetchNormalizedMarketById, type NormalizedMarket } from "./polymarketClient";
import { getAllMarkets } from "./marketDataService";
import {
  buildExecutionTruthDigest,
  estimatedNetPerCycle,
  fillPlausibility,
  isMachineObserved,
  isRobotQuoteableGate,
  observedDepth,
  observedSpread,
} from "./executionTruthEngine";

function r6(n: number): number {
  return Math.round(n * 1_000_000) / 1_000_000;
}

function sleep(ms: number): Promise<void> {
  return new Promise(res => setTimeout(res, ms));
}

function median(nums: number[]): number {
  if (nums.length === 0) return 0;
  const s = [...nums].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function stdev(nums: number[]): number {
  if (nums.length < 2) return 0;
  const mean = nums.reduce((a, b) => a + b, 0) / nums.length;
  const v = nums.reduce((s, x) => s + (x - mean) ** 2, 0) / (nums.length - 1);
  return Math.sqrt(v);
}

function midPrice(m: NormalizedMarket): number {
  if (m.prices.length < 2) return r6(m.prices[0] ?? 0);
  return r6((m.prices[0] + m.prices[1]) / 2);
}

function slipFromLiquidity(liq: number): number {
  return r6(0.0034 + 0.021 / Math.sqrt(1 + Math.max(0, liq) / 8500));
}

function executableHalfTurn(spread: number, liq: number): number {
  return r6(Math.min(0.48, spread) / 2 + slipFromLiquidity(liq));
}

export type ExecutionTruthCaptureVerdict =
  | "insufficient_execution_truth_data"
  | "execution_truth_still_too_weak"
  | "execution_truth_sufficient_for_recalibration"
  | "execution_truth_reveals_plausible_positive_cycle_zone";

export interface ExecutionTruthCaptureRow {
  marketId: string;
  marketTitle: string;
  observedSpreadSnapshot: number;
  observedDepthSnapshot: number;
  fillPlausibility: number;
  decayWindowEvidence: string;
  adverseMoveEvidence: string;
  bestObservedExecutableCost: number;
  medianObservedExecutableCost: number;
  worstObservedExecutableCost: number;
  supportingNote: string;
}

export interface StrongestCaptureRow {
  marketId: string;
  marketTitle: string;
  medianObservedExecutableCost: number;
  worstObservedExecutableCost: number;
  fillPlausibility: number;
}

export interface ExecutionTruthCaptureDigest {
  probeVersion: "execution-truth-capture-v1";
  readDisclaimer: string;
  executionTruthCaptureVerdict: ExecutionTruthCaptureVerdict;
  rowsEvaluated: number;
  rowsWithSpreadEvidence: number;
  rowsWithDepthEvidence: number;
  rowsWithDecayEvidence: number;
  rowsWithFillEvidence: number;
  strongestRows: StrongestCaptureRow[];
  executionTruthCaptureSummaryLine: string;
  rows: ExecutionTruthCaptureRow[];
  computedAt: string;
}

const SNAPSHOTS = 3;
const SNAPSHOT_GAP_MS = 280;
const MAX_MARKETS = 8;
const QUOTEABLE_PICK = 5;
const REJECTED_PICK = 3;

function shortenTitle(q: string, n = 120): string {
  return q.length > n ? `${q.slice(0, n - 1)}…` : q;
}

function pickSample(): { marketId: string; marketTitle: string; bucket: "quoteable" | "rejected" }[] {
  const digest = buildExecutionTruthDigest();
  const all = getAllMarkets();
  const out: { marketId: string; marketTitle: string; bucket: "quoteable" | "rejected" }[] = [];
  const seen = new Set<string>();

  for (const s of digest.strongestQuoteableMarkets.slice(0, QUOTEABLE_PICK)) {
    if (seen.has(s.marketId)) continue;
    seen.add(s.marketId);
    out.push({ marketId: s.marketId, marketTitle: s.marketTitle, bucket: "quoteable" });
  }

  const rejected = all
    .filter(m => isMachineObserved(m) && !isRobotQuoteableGate(m))
    .sort((a, b) => estimatedNetPerCycle(b) - estimatedNetPerCycle(a))
    .slice(0, REJECTED_PICK * 2);

  for (const m of rejected) {
    if (out.length >= MAX_MARKETS) break;
    if (seen.has(m.id)) continue;
    seen.add(m.id);
    out.push({
      marketId: m.id,
      marketTitle: shortenTitle(m.question),
      bucket: "rejected",
    });
    if (out.filter(x => x.bucket === "rejected").length >= REJECTED_PICK) break;
  }

  return out.slice(0, MAX_MARKETS);
}

async function snapshotsForMarket(marketId: string): Promise<NormalizedMarket[]> {
  const snaps: NormalizedMarket[] = [];
  for (let i = 0; i < SNAPSHOTS; i++) {
    const m = await fetchNormalizedMarketById(marketId);
    if (m) snaps.push(m);
    if (i < SNAPSHOTS - 1) await sleep(SNAPSHOT_GAP_MS);
  }
  return snaps;
}

function buildRow(
  marketId: string,
  titleHint: string,
  bucket: "quoteable" | "rejected",
  snaps: NormalizedMarket[],
): ExecutionTruthCaptureRow {
  if (snaps.length === 0) {
    return {
      marketId,
      marketTitle: titleHint,
      observedSpreadSnapshot: 0,
      observedDepthSnapshot: 0,
      fillPlausibility: 0,
      decayWindowEvidence: "snaps=0 fetch_failed_or_inactive",
      adverseMoveEvidence: "n/a",
      bestObservedExecutableCost: 0,
      medianObservedExecutableCost: 0,
      worstObservedExecutableCost: 0,
      supportingNote: `bucket=${bucket} | no_gamma_snapshots`,
    };
  }

  const spreads = snaps.map(s => observedSpread(s));
  const depths = snaps.map(s => observedDepth(s));
  const mids = snaps.map(midPrice);
  const liqs = snaps.map(s => s.liquidity);
  const spreadMean = r6(spreads.reduce((a, b) => a + b, 0) / spreads.length);
  const depthMean = r6(depths.reduce((a, b) => a + b, 0) / depths.length);
  const spreadStd = r6(stdev(spreads));
  let maxMidDelta = 0;
  for (let i = 1; i < mids.length; i++) {
    maxMidDelta = Math.max(maxMidDelta, Math.abs(mids[i] - mids[i - 1]));
  }
  const maxSpreadDelta = spreads.length >= 2 ? Math.max(...spreads.map((x, i, arr) => (i ? Math.abs(x - arr[i - 1]) : 0))) : 0;

  const costs = snaps.map(s => executableHalfTurn(s.spread, s.liquidity));

  const last = snaps[snaps.length - 1];
  const fill = fillPlausibility(last);

  const decayWindowEvidence = `snaps=${snaps.length} spread_mean=${spreadMean} spread_stdev=${spreadStd} mid_drift_max=${r6(maxMidDelta)} spread_step_max=${r6(maxSpreadDelta)}`;
  const adverseMoveEvidence = `max_abs_mid_delta=${r6(maxMidDelta)} max_spread_delta=${r6(maxSpreadDelta)}`;

  const p0 = last.prices[0] ?? 0;
  const p1 = last.prices[1] ?? 0;
  const supportingNote = `bucket=${bucket} | gamma_by_id×${snaps.length} | outcome_prices_last≈${r6(p0)}/${r6(p1)} | no_clob_book depth=liquidity_proxy`;

  return {
    marketId,
    marketTitle: shortenTitle(last.question || titleHint),
    observedSpreadSnapshot: spreadMean,
    observedDepthSnapshot: depthMean,
    fillPlausibility: fill,
    decayWindowEvidence,
    adverseMoveEvidence,
    bestObservedExecutableCost: r6(Math.min(...costs)),
    medianObservedExecutableCost: r6(median(costs)),
    worstObservedExecutableCost: r6(Math.max(...costs)),
    supportingNote,
  };
}

export async function buildExecutionTruthCaptureDigest(): Promise<ExecutionTruthCaptureDigest> {
  const sample = pickSample();
  const rows: ExecutionTruthCaptureRow[] = [];
  for (const s of sample) {
    const snaps = await snapshotsForMarket(s.marketId);
    rows.push(buildRow(s.marketId, s.marketTitle, s.bucket, snaps));
  }

  const rowsEvaluated = rows.length;
  const rowsWithSpreadEvidence = rows.filter(r => !r.decayWindowEvidence.startsWith("snaps=0")).length;
  const rowsWithDepthEvidence = rows.filter(r => r.observedDepthSnapshot > 0 && r.observedSpreadSnapshot > 0).length;
  const rowsWithDecayEvidence = rows.filter(r => {
    const m = /spread_stdev=([0-9.]+)/.exec(r.decayWindowEvidence);
    const d = /mid_drift_max=([0-9.]+)/.exec(r.decayWindowEvidence);
    const sd = m ? parseFloat(m[1]) : 0;
    const md = d ? parseFloat(d[1]) : 0;
    return sd > 1e-7 || md > 5e-5;
  }).length;
  const rowsWithFillEvidence = rows.filter(r => r.fillPlausibility >= 0.22).length;

  const strongestRows: StrongestCaptureRow[] = [...rows]
    .filter(r => r.medianObservedExecutableCost > 0)
    .sort((a, b) => a.medianObservedExecutableCost - b.medianObservedExecutableCost)
    .slice(0, 6)
    .map(r => ({
      marketId: r.marketId,
      marketTitle: r.marketTitle,
      medianObservedExecutableCost: r.medianObservedExecutableCost,
      worstObservedExecutableCost: r.worstObservedExecutableCost,
      fillPlausibility: r.fillPlausibility,
    }));

  let plausibleZone = false;
  for (const r of rows) {
    if (r.observedSpreadSnapshot <= 0) continue;
    const grossProxy = r6(r.observedSpreadSnapshot * 0.13 * r.fillPlausibility);
    const roundTripExec = r6(2 * r.medianObservedExecutableCost + 0.004);
    if (grossProxy - roundTripExec > 0.0015) plausibleZone = true;
  }

  const sufficientRecal =
    rowsWithSpreadEvidence >= 4 && rowsWithDecayEvidence >= 2 && rowsWithDepthEvidence >= 3;

  let executionTruthCaptureVerdict: ExecutionTruthCaptureVerdict;
  if (rowsEvaluated < 3 || rowsWithSpreadEvidence < 2) {
    executionTruthCaptureVerdict = "insufficient_execution_truth_data";
  } else if (plausibleZone) {
    executionTruthCaptureVerdict = "execution_truth_reveals_plausible_positive_cycle_zone";
  } else if (sufficientRecal) {
    executionTruthCaptureVerdict = "execution_truth_sufficient_for_recalibration";
  } else {
    executionTruthCaptureVerdict = "execution_truth_still_too_weak";
  }

  const executionTruthCaptureSummaryLine = `exec_truth_capture: verdict=${executionTruthCaptureVerdict} | rows=${rowsEvaluated} spread_ev=${rowsWithSpreadEvidence} depth_ev=${rowsWithDepthEvidence} decay_ev=${rowsWithDecayEvidence} fill_ev=${rowsWithFillEvidence}`;

  return {
    probeVersion: "execution-truth-capture-v1",
    readDisclaimer:
      "Capture v1: snapshots HTTP Gamma por marketId (sem CLOB). Spread/profundidade são proxies; drift entre snapshots é evidência limitada de curto horizonte. Não executa ordens. Serve só para calibrar se o modelo de custo está mais severo que a realidade observável neste canal.",
    executionTruthCaptureVerdict,
    rowsEvaluated,
    rowsWithSpreadEvidence,
    rowsWithDepthEvidence,
    rowsWithDecayEvidence,
    rowsWithFillEvidence,
    strongestRows,
    executionTruthCaptureSummaryLine,
    rows,
    computedAt: new Date().toISOString(),
  };
}
