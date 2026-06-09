/**
 * Mechanical Edge Census — runner de PARTIÇÃO / negrisk (read-only puro).
 *
 * Censo de eventos mutuamente exclusivos (negRisk) sobre a Gamma:
 *  Tier 0 (mid screen, barato): para cada evento negRisk, soma os mids YES das
 *    sub-pernas; só prossegue se Σ mid_yes < 1 (underround no mid — raro; a maioria
 *    soma > 1 pelo vig). Isso limita drasticamente os fetches de livro.
 *  Tier 1 (best ask): busca o livro YES de cada perna, e marca se Σ best_ask_yes
 *    < 1 − minGross (comprar a cesta inteira < payout garantido 1).
 *  Tier 2 (fundo): VWAP por profundidade no tamanho-alvo, evaluateMecBasket (6
 *    custos), e appenda no ledger JSONL.
 *
 * Diferente do binário (que cruza 2 spreads do MESMO livro e é estruturalmente
 * vazio), a partição compra YES de k livros independentes — a soma PODE ficar < 1.
 *
 * Apenas leituras: GET Gamma /events + GET CLOB /book. Sem ordens, sem paper, sem
 * .paper, sem microcapital, sem execução.
 *
 * Variáveis de ambiente:
 *   MEC_PART_LEDGER_PATH    (default: $HOME/mechanical-edge-census-partition-history.jsonl)
 *   MEC_TARGET_SIZE_USD     (default: 100)
 *   MEC_MIN_GROSS           (default: 0.003)
 *   MEC_PART_EVENT_LIMIT    (default: 300 eventos varridos)
 *   MEC_PART_MAX_LEGS       (default: 24 — pula partições gigantes)
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
  partitionUnderroundFlag,
  targetSharesPerLeg,
  classifyResolutionCategory,
} from "../lib/mechanicalEdgeCensusBook";
import {
  fetchRawBook,
  assertSafeLedgerPath,
  appendMecLedger,
} from "../lib/mechanicalEdgeCensusFetch";
import {
  loadNegRiskEventsPage,
  extractPartitionEvent,
  MEC_PARTITION_PAGE_LIMIT,
} from "../lib/mecPartitionScan";

const PAGE_LIMIT = MEC_PARTITION_PAGE_LIMIT;

const LEDGER_PATH =
  process.env.MEC_PART_LEDGER_PATH ??
  path.join(process.env.HOME ?? ".", "mechanical-edge-census-partition-history.jsonl");
const TARGET_SIZE_USD = parseFloat(process.env.MEC_TARGET_SIZE_USD ?? "100") || 100;
const MIN_GROSS = parseFloat(process.env.MEC_MIN_GROSS ?? "0.003") || 0.003;
const EVENT_LIMIT = parseInt(process.env.MEC_PART_EVENT_LIMIT ?? "300", 10) || 300;
const MAX_LEGS = parseInt(process.env.MEC_PART_MAX_LEGS ?? "24", 10) || 24;

function daysToResolution(endDate: unknown, now: Date): number {
  if (typeof endDate !== "string" || !endDate) return 0;
  const t = new Date(endDate).getTime();
  if (!Number.isFinite(t)) return 0;
  return Math.max(0, (t - now.getTime()) / 86_400_000);
}

async function runCensus(): Promise<void> {
  assertSafeLedgerPath(LEDGER_PATH);
  const now = new Date();
  const capturedAt = now.toISOString();

  process.stdout.write(
    `[mec-census-partition] starting version=${MEC_VERSION} target_usd=${TARGET_SIZE_USD} min_gross=${MIN_GROSS} event_limit=${EVENT_LIMIT} max_legs=${MAX_LEGS} ledger=${LEDGER_PATH}\n`,
  );

  let eventsScanned = 0;
  let negRiskEvents = 0;
  let tier0MidUnderround = 0;
  let tier1Flagged = 0;
  let tier2Persisted = 0;
  const verdictTally: Record<string, number> = {};

  for (let offset = 0; offset < EVENT_LIMIT; offset += PAGE_LIMIT) {
    let page: Record<string, unknown>[];
    try {
      page = await loadNegRiskEventsPage(offset);
    } catch (err) {
      process.stderr.write(
        `[mec-census-partition] gamma_events_failed offset=${offset}: ${err instanceof Error ? err.message : String(err)}\n`,
      );
      break;
    }
    if (page.length === 0) break;

    for (const ev of page) {
      eventsScanned++;
      const pe = extractPartitionEvent(ev);
      if (!pe) continue;
      negRiskEvents++;
      if (pe.legs.length > MAX_LEGS) continue;

      /** Tier 0 — mid screen barato: ask sum ≥ mid sum, então se mid sum ≥ 1, ask sum ≥ 1 (sem chance). */
      const sumMidYes = pe.legs.reduce((a, l) => a + l.yesMid, 0);
      if (sumMidYes >= 1) continue;
      tier0MidUnderround++;

      /** Tier 1 — best ask real de cada perna. */
      const books = await Promise.all(pe.legs.map(l => fetchRawBook(l.yesToken)));
      const bestAsks: number[] = [];
      let bookMissing = false;
      for (const b of books) {
        if (!b) {
          bookMissing = true;
          break;
        }
        const ba = bestPrice(b.asks, "buy");
        if (ba === null) {
          bookMissing = true;
          break;
        }
        bestAsks.push(ba);
      }
      if (bookMissing || bestAsks.length !== pe.legs.length) continue;
      if (!partitionUnderroundFlag(bestAsks, MIN_GROSS)) continue;
      tier1Flagged++;

      /** Tier 2 — VWAP por profundidade no tamanho-alvo. */
      const capitalPerUnitBest = bestAsks.reduce((a, b) => a + b, 0);
      const shares = targetSharesPerLeg(TARGET_SIZE_USD, capitalPerUnitBest);
      const legs: MecLeg[] = [];
      let vwapMissing = false;
      for (let i = 0; i < pe.legs.length; i++) {
        const v = computeVwap(books[i]!.asks, shares, "buy");
        if (v.vwap === null) {
          vwapMissing = true;
          break;
        }
        const bb = bestPrice(books[i]!.bids, "sell");
        const spread = bb !== null ? Math.max(0, bestAsks[i]! - bb) : 0;
        legs.push({
          marketId: `${pe.legs[i]!.marketId}:YES`,
          side: "buy",
          vwapPrice: v.vwap,
          bestPrice: bestAsks[i]!,
          depthTop3: depthTopN(books[i]!.asks, 3, "buy"),
          spread,
        });
      }
      if (vwapMissing) continue;

      /** Probabilidade implícita do "campo" não enumerado (proxy de não-exaustividade). */
      const sumBest = bestAsks.reduce((a, b) => a + b, 0);
      const fieldProbabilityEstimate = Math.max(0, Math.round((1 - sumBest) * 1e6) / 1e6);

      const category = classifyResolutionCategory(pe.title);
      const edgeType = pe.conversionFeeFrac > 0 ? "NEGRISK_CONVERSION" : "PARTITION_UNDERROUND";
      const evaluation = evaluateMecBasket(
        {
          legs,
          edgeType,
          daysToResolution: daysToResolution(pe.endDate, now),
          category,
          conversionFeeFrac: pe.conversionFeeFrac,
        },
        { ...MEC_DEFAULT_COST_MODEL, targetSizeUsd: TARGET_SIZE_USD },
      );

      verdictTally[evaluation.verdict] = (verdictTally[evaluation.verdict] ?? 0) + 1;

      appendMecLedger(LEDGER_PATH, {
        timestamp: capturedAt,
        mecVersion: MEC_VERSION,
        eventId: pe.eventId,
        title: pe.title.slice(0, 180),
        category,
        edgeType,
        k: pe.legs.length,
        conversionFeeFrac: pe.conversionFeeFrac,
        fieldProbabilityEstimate,
        evaluation,
        canUseForExecution: false,
        dedupeKey: `${pe.eventId}|${edgeType}|${capturedAt.slice(0, 13)}|${MEC_VERSION}`,
      });
      tier2Persisted++;
      process.stdout.write(
        `[mec-census-partition] flagged eventId=${pe.eventId} k=${pe.legs.length} cat=${category} type=${edgeType} gross=${evaluation.grossEdge} net=${evaluation.netEdge} verdict=${evaluation.verdict}\n`,
      );
    }
  }

  process.stdout.write(
    `[mec-census-partition] ${capturedAt} eventsScanned=${eventsScanned} negRiskEvents=${negRiskEvents} tier0MidUnderround=${tier0MidUnderround} tier1Flagged=${tier1Flagged} tier2Persisted=${tier2Persisted} verdicts=${JSON.stringify(verdictTally)}\n`,
  );
  process.stdout.write("[mec-census-partition] exit_ok\n");
}

runCensus().catch(err => {
  process.stderr.write(
    `[mec-census-partition] fatal: ${err instanceof Error ? err.message : String(err)}\n`,
  );
  process.exitCode = 1;
});
