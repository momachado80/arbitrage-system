/**
 * Mechanical Edge Census — runner Tier 1+2 (read-only puro).
 *
 * Censo binário YES+NO sobre o universo Gamma:
 *  Tier 1 (barato): para cada mercado binário, lê o melhor ask dos tokens YES e NO
 *    do CLOB e marca os que têm ask_yes + ask_no < 1 − minGross (underround).
 *  Tier 2 (fundo): para cada flag, computa VWAP por profundidade no tamanho-alvo,
 *    monta as pernas, chama evaluateMecBasket (6 custos) e appenda no ledger JSONL.
 *
 * Apenas leituras: GET na Gamma (markets) + GET no CLOB (book). Sem ordens, sem
 * paper engine, sem .paper, sem microcapital, sem worker de execução, sem envio
 * para rede de execução. Escrita append-only no JSONL do volume.
 *
 * Variáveis de ambiente:
 *   MEC_LEDGER_PATH        (default: $HOME/mechanical-edge-census-history.jsonl)
 *   MEC_TARGET_SIZE_USD    (default: 100)
 *   MEC_MIN_GROSS          (default: 0.003)
 *   MEC_PLAN_LIMIT         (default: 200 mercados varridos)
 */

import path from "path";

import {
  evaluateMecBasket,
  MEC_DEFAULT_COST_MODEL,
  MEC_VERSION,
  type MecLeg,
} from "../lib/mechanicalEdgeCensus";
import {
  computeVwap,
  bestPrice,
  depthTopN,
  binaryUnderroundFlag,
  targetSharesPerLeg,
  classifyResolutionCategory,
} from "../lib/mechanicalEdgeCensusBook";
import {
  fetchJson,
  fetchRawBook,
  assertSafeLedgerPath,
  appendMecLedger,
  jsonArray,
} from "../lib/mechanicalEdgeCensusFetch";

const GAMMA_LIST = "https://gamma-api.polymarket.com/markets";
const PAGE_LIMIT = 100;
const GAMMA_HTTP_MS = 12_000;

const LEDGER_PATH =
  process.env.MEC_LEDGER_PATH ??
  path.join(process.env.HOME ?? ".", "mechanical-edge-census-history.jsonl");
const TARGET_SIZE_USD = parseFloat(process.env.MEC_TARGET_SIZE_USD ?? "100") || 100;
const MIN_GROSS = parseFloat(process.env.MEC_MIN_GROSS ?? "0.003") || 0.003;
const SCAN_LIMIT = parseInt(process.env.MEC_PLAN_LIMIT ?? "200", 10) || 200;

async function loadGammaMarketsPage(offset: number): Promise<Record<string, unknown>[]> {
  const url = `${GAMMA_LIST}?active=true&closed=false&limit=${PAGE_LIMIT}&offset=${offset}`;
  const body = await fetchJson(url, GAMMA_HTTP_MS);
  return Array.isArray(body) ? (body as Record<string, unknown>[]) : [];
}

function daysToResolution(endDate: unknown, now: Date): number {
  if (typeof endDate !== "string" || !endDate) return 0;
  const t = new Date(endDate).getTime();
  if (!Number.isFinite(t)) return 0;
  return Math.max(0, (t - now.getTime()) / 86_400_000);
}

interface BinaryCandidate {
  marketId: string;
  question: string;
  endDate: unknown;
  yesToken: string;
  noToken: string;
}

function extractBinaryCandidate(row: Record<string, unknown>): BinaryCandidate | null {
  const outcomes = jsonArray(row.outcomes);
  const tokens = jsonArray(row.clobTokenIds);
  if (outcomes.length !== 2 || tokens.length !== 2) return null;
  const yesIdx = outcomes.findIndex(o => /^yes$/i.test(o.trim()));
  const yesToken = yesIdx >= 0 ? tokens[yesIdx]! : tokens[0]!;
  const noToken = yesIdx >= 0 ? tokens[1 - yesIdx]! : tokens[1]!;
  const marketId = row.id != null ? String(row.id) : "";
  if (!marketId || !yesToken || !noToken) return null;
  const question = typeof row.question === "string" ? row.question : typeof row.slug === "string" ? row.slug : "";
  return { marketId, question, endDate: row.endDate ?? row.end_date ?? null, yesToken, noToken };
}

async function runCensus(): Promise<void> {
  assertSafeLedgerPath(LEDGER_PATH);
  const now = new Date();
  const capturedAt = now.toISOString();

  process.stdout.write(
    `[mec-census] starting version=${MEC_VERSION} target_usd=${TARGET_SIZE_USD} min_gross=${MIN_GROSS} scan_limit=${SCAN_LIMIT} ledger=${LEDGER_PATH}\n`,
  );

  let scanned = 0;
  let binaryEligible = 0;
  let tier1Flagged = 0;
  let tier2Persisted = 0;
  const verdictTally: Record<string, number> = {};

  for (let offset = 0; offset < SCAN_LIMIT; offset += PAGE_LIMIT) {
    let page: Record<string, unknown>[];
    try {
      page = await loadGammaMarketsPage(offset);
    } catch (err) {
      process.stderr.write(`[mec-census] gamma_page_failed offset=${offset}: ${err instanceof Error ? err.message : String(err)}\n`);
      break;
    }
    if (page.length === 0) break;

    for (const row of page) {
      scanned++;
      const cand = extractBinaryCandidate(row);
      if (!cand) continue;
      binaryEligible++;

      const [yesBook, noBook] = await Promise.all([fetchRawBook(cand.yesToken), fetchRawBook(cand.noToken)]);
      if (!yesBook || !noBook) continue;

      const bestAskYes = bestPrice(yesBook.asks, "buy");
      const bestAskNo = bestPrice(noBook.asks, "buy");
      if (bestAskYes === null || bestAskNo === null) continue;

      /** Tier 1 — flag barato. */
      if (!binaryUnderroundFlag(bestAskYes, bestAskNo, MIN_GROSS)) continue;
      tier1Flagged++;

      /** Tier 2 — VWAP no tamanho-alvo + 6 custos. */
      const capitalPerUnitBest = bestAskYes + bestAskNo;
      const shares = targetSharesPerLeg(TARGET_SIZE_USD, capitalPerUnitBest);
      const yesV = computeVwap(yesBook.asks, shares, "buy");
      const noV = computeVwap(noBook.asks, shares, "buy");
      if (yesV.vwap === null || noV.vwap === null) continue;

      const legs: MecLeg[] = [
        {
          marketId: `${cand.marketId}:YES`,
          side: "buy",
          vwapPrice: yesV.vwap,
          bestPrice: bestAskYes,
          depthTop3: depthTopN(yesBook.asks, 3, "buy"),
          spread: 0,
        },
        {
          marketId: `${cand.marketId}:NO`,
          side: "buy",
          vwapPrice: noV.vwap,
          bestPrice: bestAskNo,
          depthTop3: depthTopN(noBook.asks, 3, "buy"),
          spread: 0,
        },
      ];

      const category = classifyResolutionCategory(cand.question);
      const evaluation = evaluateMecBasket(
        {
          legs,
          edgeType: "BINARY_UNDERROUND",
          daysToResolution: daysToResolution(cand.endDate, now),
          category,
        },
        { ...MEC_DEFAULT_COST_MODEL, targetSizeUsd: TARGET_SIZE_USD },
      );

      verdictTally[evaluation.verdict] = (verdictTally[evaluation.verdict] ?? 0) + 1;

      const entry = {
        timestamp: capturedAt,
        mecVersion: MEC_VERSION,
        marketId: cand.marketId,
        question: cand.question.slice(0, 180),
        category,
        edgeType: "BINARY_UNDERROUND",
        evaluation,
        canUseForExecution: false,
        dedupeKey: `${cand.marketId}|BINARY_UNDERROUND|${capturedAt.slice(0, 13)}|${MEC_VERSION}`,
      };
      appendMecLedger(LEDGER_PATH, entry);
      tier2Persisted++;
      process.stdout.write(
        `[mec-census] flagged marketId=${cand.marketId} cat=${category} gross=${evaluation.grossEdge} net=${evaluation.netEdge} verdict=${evaluation.verdict}\n`,
      );
    }
  }

  process.stdout.write(
    `[mec-census] ${capturedAt} scanned=${scanned} binaryEligible=${binaryEligible} tier1Flagged=${tier1Flagged} tier2Persisted=${tier2Persisted} verdicts=${JSON.stringify(verdictTally)}\n`,
  );
  process.stdout.write("[mec-census] exit_ok\n");
}

runCensus().catch(err => {
  process.stderr.write(`[mec-census] fatal: ${err instanceof Error ? err.message : String(err)}\n`);
  process.exitCode = 1;
});
