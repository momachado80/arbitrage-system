/**
 * Resolution Convergence Census — runner read-only (filão #4).
 *
 * Surfa mercados binários PERTO da resolução negociando PERTO de um extremo (um
 * lado fortemente favorito) e mede o desconto de convergência líquido de custos
 * mecânicos. NÃO declara edge: convergência exige julgamento de fair-value
 * (o desconto pode ser risco real residual, não mispricing). Verdict de cada
 * candidato é "convergence_candidate_needs_fair_value" — para revisão humana.
 *
 * Apenas leituras: GET Gamma /markets + GET CLOB /book. Sem ordens, sem paper,
 * sem .paper, sem microcapital, sem execução.
 *
 * Env:
 *   CONV_LEDGER_PATH      (default: $HOME/resolution-convergence-history.jsonl)
 *   CONV_TARGET_SIZE_USD  (default: 100)
 *   CONV_MAX_DAYS         (default: 7 — só mercados resolvendo dentro disso)
 *   CONV_SCAN_LIMIT       (default: 1000 mercados varridos)
 */

import path from "path";

import {
  scoreConvergenceCandidate,
  CONV_NEAR_EXTREME_MID,
  CONV_NEAR_RESOLUTION_DAYS,
  type ConvergenceCostModel,
} from "../lib/resolutionConvergence";
import { classifyResolutionCategory, bestPrice } from "../lib/mechanicalEdgeCensusBook";
import { MEC_DEFAULT_COST_MODEL } from "../lib/mechanicalEdgeCensus";
import {
  fetchJson,
  fetchRawBook,
  assertSafeLedgerPath,
  appendMecLedger,
  jsonArray,
  jsonNumberArray,
} from "../lib/mechanicalEdgeCensusFetch";

const GAMMA_LIST = "https://gamma-api.polymarket.com/markets";
const PAGE_LIMIT = 100;
const GAMMA_HTTP_MS = 12_000;

const LEDGER_PATH =
  process.env.CONV_LEDGER_PATH ??
  path.join(process.env.HOME ?? ".", "resolution-convergence-history.jsonl");
const TARGET_SIZE_USD = parseFloat(process.env.CONV_TARGET_SIZE_USD ?? "100") || 100;
const MAX_DAYS = parseFloat(process.env.CONV_MAX_DAYS ?? String(CONV_NEAR_RESOLUTION_DAYS)) || CONV_NEAR_RESOLUTION_DAYS;
const SCAN_LIMIT = parseInt(process.env.CONV_SCAN_LIMIT ?? "1000", 10) || 1000;

const COST_MODEL: ConvergenceCostModel = {
  costOfCapitalAnnual: MEC_DEFAULT_COST_MODEL.costOfCapitalAnnual,
  gasPerTxUsd: MEC_DEFAULT_COST_MODEL.gasPerTxUsd,
  targetSizeUsd: TARGET_SIZE_USD,
  umaHaircutByCategory: MEC_DEFAULT_COST_MODEL.umaHaircutByCategory,
};

function num(x: unknown): number {
  const n = typeof x === "number" ? x : parseFloat(String(x));
  return Number.isFinite(n) ? n : NaN;
}

function daysToResolution(endDate: unknown, now: Date): number {
  if (typeof endDate !== "string" || !endDate) return Infinity;
  const t = new Date(endDate).getTime();
  if (!Number.isFinite(t)) return Infinity;
  return (t - now.getTime()) / 86_400_000;
}

interface BinaryRow {
  marketId: string;
  question: string;
  endDate: unknown;
  yesToken: string;
  noToken: string;
  yesMidHint: number;
}

function extractBinary(row: Record<string, unknown>): BinaryRow | null {
  const outcomes = jsonArray(row.outcomes);
  const tokens = jsonArray(row.clobTokenIds);
  if (outcomes.length !== 2 || tokens.length !== 2) return null;
  const yesIdx = outcomes.findIndex(o => /^yes$/i.test(o.trim()));
  const idx = yesIdx >= 0 ? yesIdx : 0;
  const yesToken = tokens[idx]!;
  const noToken = tokens[1 - idx]!;
  const marketId = row.id != null ? String(row.id) : "";
  if (!marketId || !yesToken || !noToken) return null;
  const prices = jsonNumberArray(row.outcomePrices);
  const yesMidHint = prices.length > idx ? prices[idx]! : NaN;
  const question = typeof row.question === "string" ? row.question : typeof row.slug === "string" ? (row.slug as string) : "";
  return { marketId, question, endDate: row.endDate ?? row.end_date ?? null, yesToken, noToken, yesMidHint };
}

async function loadGammaMarketsPage(offset: number, now: Date): Promise<Record<string, unknown>[]> {
  /** Filtro server-side pela JANELA de resolução. Ordenar por endDate ascendente
   *  traz primeiro mercados vencidos-mas-abertos (zumbis aguardando UMA), que
   *  soterram os near-term além do scan limit. A janela end_date_min/max corta
   *  isso na origem. */
  const minIso = new Date(now.getTime() - 60_000).toISOString();
  const maxIso = new Date(now.getTime() + MAX_DAYS * 86_400_000).toISOString();
  const url =
    `${GAMMA_LIST}?active=true&closed=false&limit=${PAGE_LIMIT}&offset=${offset}` +
    `&end_date_min=${encodeURIComponent(minIso)}&end_date_max=${encodeURIComponent(maxIso)}`;
  const body = await fetchJson(url, GAMMA_HTTP_MS);
  return Array.isArray(body) ? (body as Record<string, unknown>[]) : [];
}

async function runCensus(): Promise<void> {
  assertSafeLedgerPath(LEDGER_PATH);
  const now = new Date();
  const capturedAt = now.toISOString();

  process.stdout.write(
    `[conv-census] starting target_usd=${TARGET_SIZE_USD} max_days=${MAX_DAYS} near_extreme=${CONV_NEAR_EXTREME_MID} scan_limit=${SCAN_LIMIT} ledger=${LEDGER_PATH}\n`,
  );

  let scanned = 0;
  let binaryEligible = 0;
  let nearResolution = 0;
  let nearExtreme = 0;
  let candidates = 0;
  const verdictTally: Record<string, number> = {};

  for (let offset = 0; offset < SCAN_LIMIT; offset += PAGE_LIMIT) {
    let page: Record<string, unknown>[];
    try {
      page = await loadGammaMarketsPage(offset, now);
    } catch (err) {
      process.stderr.write(`[conv-census] gamma_page_failed offset=${offset}: ${err instanceof Error ? err.message : String(err)}\n`);
      break;
    }
    if (page.length === 0) break;

    for (const row of page) {
      scanned++;
      const b = extractBinary(row);
      if (!b) continue;
      binaryEligible++;

      const days = daysToResolution(b.endDate, now);
      if (!(days <= MAX_DAYS) || days < 0) continue;
      nearResolution++;

      /** Pré-screen barato pelo mid hint da Gamma antes de tocar o livro. */
      if (Number.isFinite(b.yesMidHint)) {
        if (b.yesMidHint < CONV_NEAR_EXTREME_MID && b.yesMidHint > 1 - CONV_NEAR_EXTREME_MID) continue;
      }

      const [yesBook, noBook] = await Promise.all([fetchRawBook(b.yesToken), fetchRawBook(b.noToken)]);
      if (!yesBook || !noBook) continue;
      const yesAsk = bestPrice(yesBook.asks, "buy");
      const noAsk = bestPrice(noBook.asks, "buy");
      const yesBid = bestPrice(yesBook.bids, "sell");
      if (yesAsk === null || noAsk === null) continue;
      /** Mid do YES via livro (fallback ao hint da Gamma). */
      const yesMid = yesBid !== null ? (yesAsk + yesBid) / 2 : Number.isFinite(b.yesMidHint) ? b.yesMidHint : yesAsk;

      const score = scoreConvergenceCandidate(
        { yesMid, yesAsk, noAsk, daysToResolution: days, category: classifyResolutionCategory(b.question) },
        COST_MODEL,
      );

      if (score.favoredSide !== null && score.verdict !== "not_near_extreme") nearExtreme++;
      verdictTally[score.verdict] = (verdictTally[score.verdict] ?? 0) + 1;

      if (score.verdict !== "convergence_candidate_needs_fair_value") continue;
      candidates++;

      appendMecLedger(LEDGER_PATH, {
        timestamp: capturedAt,
        probe: "resolution_convergence_census_v1",
        marketId: b.marketId,
        question: b.question.slice(0, 180),
        category: classifyResolutionCategory(b.question),
        score,
        canUseForExecution: false,
        dedupeKey: `${b.marketId}|CONV|${capturedAt.slice(0, 13)}`,
      });
      process.stdout.write(
        `[conv-census] candidate marketId=${b.marketId} favored=${score.favoredSide} discount=${score.discount} net=${score.netDiscountAfterCosts} days=${score.daysToResolution.toFixed(2)} anchorable=${score.anchorable}\n`,
      );
    }
  }

  process.stdout.write(
    `[conv-census] ${capturedAt} scanned=${scanned} binaryEligible=${binaryEligible} nearResolution=${nearResolution} nearExtreme=${nearExtreme} candidates=${candidates} verdicts=${JSON.stringify(verdictTally)}\n`,
  );
  process.stdout.write("[conv-census] exit_ok\n");
}

runCensus().catch(err => {
  process.stderr.write(`[conv-census] fatal: ${err instanceof Error ? err.message : String(err)}\n`);
  process.exitCode = 1;
});
