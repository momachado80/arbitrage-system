/**
 * MEC-3 — Persistence Sampler (read-only puro).
 *
 * Mede a variável que falta para precificar "micro-edge recorrente":
 * um gap flagged num snapshot SOBREVIVE minutos depois, ou evapora?
 *
 * Fase 1 (descoberta): roda Tier 0+1 de partição negRisk (Σ best_ask < 1−minGross)
 *   e de overround binário (Σ best_bid > 1+minGross) e coleta até MAX_TRACKED cestas.
 * Fase 2 (re-observação): em offsets configuráveis (default +60s, +300s, +900s),
 *   re-busca os livros de cada cesta e recomputa o gap no melhor nível e no VWAP
 *   para tamanhos $10/$30/$100, registrando capacidade aproximada.
 * Fase 3 (verdito): summarizePersistence por cesta → persistent / decayed / transient.
 *
 * Tudo GET + append-only no ledger JSONL. Sem ordens, sem paper, sem .paper,
 * sem microcapital, sem execução. Runtime total ≈ último offset (default ~15 min).
 *
 * Env:
 *   MEC3_LEDGER_PATH     (default: $HOME/mec-persistence-history.jsonl)
 *   MEC3_INTERVALS_SEC   (default: "60,300,900")
 *   MEC3_MAX_TRACKED     (default: 8 cestas)
 *   MEC3_MIN_GROSS       (default: 0.003)
 *   MEC3_PART_EVENT_LIMIT (default: 300)
 *   MEC3_BIN_SCAN_LIMIT   (default: 300 mercados binários p/ overround)
 */

import path from "path";

import {
  computeVwap,
  bestPrice,
  depthTopN,
  partitionUnderroundFlag,
  binaryOverroundFlag,
} from "../lib/mechanicalEdgeCensusBook";
import {
  fetchJson,
  fetchRawBook,
  assertSafeLedgerPath,
  appendMecLedger,
  jsonArray,
} from "../lib/mechanicalEdgeCensusFetch";
import { loadNegRiskEventsPage, extractPartitionEvent } from "../lib/mecPartitionScan";
import {
  summarizePersistence,
  type PersistenceObservation,
} from "../lib/mecPersistence";

const GAMMA_LIST = "https://gamma-api.polymarket.com/markets";
const GAMMA_HTTP_MS = 12_000;
const MARKETS_PAGE = 100;

const LEDGER_PATH =
  process.env.MEC3_LEDGER_PATH ??
  path.join(process.env.HOME ?? ".", "mec-persistence-history.jsonl");
const INTERVALS_SEC = (process.env.MEC3_INTERVALS_SEC ?? "60,300,900")
  .split(",")
  .map(s => parseInt(s.trim(), 10))
  .filter(n => Number.isFinite(n) && n > 0)
  .sort((a, b) => a - b);
const MAX_TRACKED = parseInt(process.env.MEC3_MAX_TRACKED ?? "8", 10) || 8;
const MIN_GROSS = parseFloat(process.env.MEC3_MIN_GROSS ?? "0.003") || 0.003;
const PART_EVENT_LIMIT = parseInt(process.env.MEC3_PART_EVENT_LIMIT ?? "300", 10) || 300;
const BIN_SCAN_LIMIT = parseInt(process.env.MEC3_BIN_SCAN_LIMIT ?? "300", 10) || 300;
const VWAP_SIZES_USD = [10, 30, 100];
const MAX_LEGS = 24;

interface BasketLeg {
  tokenId: string;
  label: string;
}

interface TrackedBasket {
  basketId: string;
  kind: "PARTITION_BUY" | "BINARY_MINT_SELL";
  title: string;
  legs: BasketLeg[];
  observations: PersistenceObservation[];
}

function sleep(ms: number): Promise<void> {
  return new Promise(r => setTimeout(r, ms));
}

function r6(n: number): number {
  return Math.round(n * 1_000_000) / 1_000_000;
}

/**
 * Observa uma cesta agora: gap no melhor nível, gap no VWAP por tamanho e
 * capacidade aproximada (unidades preenchíveis no top3 × capital por unidade).
 */
async function observeBasket(
  basket: TrackedBasket,
  offsetSec: number,
): Promise<PersistenceObservation | null> {
  const side = basket.kind === "PARTITION_BUY" ? "buy" : "sell";
  const books = await Promise.all(basket.legs.map(l => fetchRawBook(l.tokenId)));
  const bests: number[] = [];
  const depths: number[] = [];
  for (const b of books) {
    if (!b) return null;
    const levels = side === "buy" ? b.asks : b.bids;
    const bp = bestPrice(levels, side);
    if (bp === null) return null;
    bests.push(bp);
    depths.push(depthTopN(levels, 3, side));
  }
  const sumBest = bests.reduce((a, b) => a + b, 0);
  const grossAtBest = r6(side === "buy" ? 1 - sumBest : sumBest - 1);
  const capitalPerUnit = side === "buy" ? sumBest : 1;

  const grossAtVwapBySize: Record<string, number | null> = {};
  for (const usd of VWAP_SIZES_USD) {
    const shares = capitalPerUnit > 0 ? usd / capitalPerUnit : 0;
    let sumVwap = 0;
    let ok = true;
    for (let i = 0; i < basket.legs.length; i++) {
      const levels = side === "buy" ? books[i]!.asks : books[i]!.bids;
      const v = computeVwap(levels, shares, side);
      if (v.vwap === null || !v.fullyFilled) {
        ok = false;
        break;
      }
      sumVwap += v.vwap;
    }
    grossAtVwapBySize[String(usd)] = ok
      ? r6(side === "buy" ? 1 - sumVwap : sumVwap - 1)
      : null;
  }

  /** Capacidade ≈ unidades preenchíveis no top3 da perna mais rasa × capital/unidade. */
  const fillableUsdApprox = r6(Math.min(...depths) * capitalPerUnit);

  return { offsetSec, grossAtBest, grossAtVwapBySize, fillableUsdApprox };
}

async function discoverPartitionBaskets(): Promise<TrackedBasket[]> {
  const out: TrackedBasket[] = [];
  for (let offset = 0; offset < PART_EVENT_LIMIT && out.length < MAX_TRACKED; offset += 80) {
    let page: Record<string, unknown>[];
    try {
      page = await loadNegRiskEventsPage(offset);
    } catch {
      break;
    }
    if (page.length === 0) break;
    for (const ev of page) {
      if (out.length >= MAX_TRACKED) break;
      const pe = extractPartitionEvent(ev);
      if (!pe || pe.legs.length > MAX_LEGS) continue;
      const sumMid = pe.legs.reduce((a, l) => a + l.yesMid, 0);
      if (sumMid >= 1) continue;
      const books = await Promise.all(pe.legs.map(l => fetchRawBook(l.yesToken)));
      const asks: number[] = [];
      let missing = false;
      for (const b of books) {
        const ba = b ? bestPrice(b.asks, "buy") : null;
        if (ba === null) {
          missing = true;
          break;
        }
        asks.push(ba);
      }
      if (missing || !partitionUnderroundFlag(asks, MIN_GROSS)) continue;
      out.push({
        basketId: `part:${pe.eventId}`,
        kind: "PARTITION_BUY",
        title: pe.title.slice(0, 120),
        legs: pe.legs.map(l => ({ tokenId: l.yesToken, label: l.marketId })),
        observations: [],
      });
    }
  }
  return out;
}

async function discoverOverroundBaskets(slots: number): Promise<TrackedBasket[]> {
  const out: TrackedBasket[] = [];
  for (let offset = 0; offset < BIN_SCAN_LIMIT && out.length < slots; offset += MARKETS_PAGE) {
    let page: Record<string, unknown>[];
    try {
      const url = `${GAMMA_LIST}?active=true&closed=false&limit=${MARKETS_PAGE}&offset=${offset}`;
      const body = await fetchJson(url, GAMMA_HTTP_MS);
      page = Array.isArray(body) ? (body as Record<string, unknown>[]) : [];
    } catch {
      break;
    }
    if (page.length === 0) break;
    for (const row of page) {
      if (out.length >= slots) break;
      const outcomes = jsonArray(row.outcomes);
      const tokens = jsonArray(row.clobTokenIds);
      if (outcomes.length !== 2 || tokens.length !== 2) continue;
      const yesIdx = outcomes.findIndex(o => /^yes$/i.test(o.trim()));
      const idx = yesIdx >= 0 ? yesIdx : 0;
      const yesToken = tokens[idx]!;
      const noToken = tokens[1 - idx]!;
      if (!yesToken || !noToken) continue;
      const [yb, nb] = await Promise.all([fetchRawBook(yesToken), fetchRawBook(noToken)]);
      if (!yb || !nb) continue;
      const bidYes = bestPrice(yb.bids, "sell");
      const bidNo = bestPrice(nb.bids, "sell");
      if (bidYes === null || bidNo === null) continue;
      if (!binaryOverroundFlag(bidYes, bidNo, MIN_GROSS)) continue;
      const q = typeof row.question === "string" ? row.question : String(row.id ?? "");
      out.push({
        basketId: `over:${String(row.id ?? "")}`,
        kind: "BINARY_MINT_SELL",
        title: q.slice(0, 120),
        legs: [
          { tokenId: yesToken, label: "YES" },
          { tokenId: noToken, label: "NO" },
        ],
        observations: [],
      });
    }
  }
  return out;
}

async function main(): Promise<void> {
  assertSafeLedgerPath(LEDGER_PATH);
  const startedAt = new Date().toISOString();
  process.stdout.write(
    `[mec3] starting intervals=${INTERVALS_SEC.join(",")}s max_tracked=${MAX_TRACKED} min_gross=${MIN_GROSS} ledger=${LEDGER_PATH}\n`,
  );

  /** Fase 1 — descoberta. */
  const partition = await discoverPartitionBaskets();
  const overround = await discoverOverroundBaskets(Math.max(0, MAX_TRACKED - partition.length));
  const baskets = [...partition, ...overround];
  process.stdout.write(
    `[mec3] discovery: partition=${partition.length} overround=${overround.length} tracked=${baskets.length}\n`,
  );
  if (baskets.length === 0) {
    process.stdout.write(
      `[mec3] NENHUMA cesta flagged no snapshot de descoberta — nada para rastrear. ` +
        `Isso por si só é dado: frequência de gaps ≈ 0 neste instante.\n[mec3] exit_ok\n`,
    );
    return;
  }
  for (const b of baskets) {
    process.stdout.write(`[mec3] tracking ${b.basketId} kind=${b.kind} k=${b.legs.length} "${b.title}"\n`);
  }

  /** Fase 2 — observação t=0 + offsets. */
  const offsets = [0, ...INTERVALS_SEC];
  const t0 = Date.now();
  for (const off of offsets) {
    const waitMs = t0 + off * 1000 - Date.now();
    if (waitMs > 0) {
      process.stdout.write(`[mec3] sleeping ${(waitMs / 1000).toFixed(0)}s until t+${off}s\n`);
      await sleep(waitMs);
    }
    for (const b of baskets) {
      const obs = await observeBasket(b, off);
      if (obs) {
        b.observations.push(obs);
        appendMecLedger(LEDGER_PATH, {
          timestamp: new Date().toISOString(),
          probe: "mec_persistence_v1",
          basketId: b.basketId,
          kind: b.kind,
          title: b.title,
          observation: obs,
          canUseForExecution: false,
        });
        process.stdout.write(
          `[mec3] t+${off}s ${b.basketId} grossBest=${obs.grossAtBest} vwap=${JSON.stringify(obs.grossAtVwapBySize)} fillable≈$${obs.fillableUsdApprox}\n`,
        );
      } else {
        process.stdout.write(`[mec3] t+${off}s ${b.basketId} book_unavailable\n`);
      }
    }
  }

  /** Fase 3 — verdito por cesta. */
  process.stdout.write(`\n[mec3] ===== SUMÁRIO DE PERSISTÊNCIA =====\n`);
  const verdictTally: Record<string, number> = {};
  for (const b of baskets) {
    const s = summarizePersistence(b.observations, MIN_GROSS);
    verdictTally[s.verdict] = (verdictTally[s.verdict] ?? 0) + 1;
    appendMecLedger(LEDGER_PATH, {
      timestamp: new Date().toISOString(),
      probe: "mec_persistence_v1",
      basketId: b.basketId,
      kind: b.kind,
      title: b.title,
      summary: s,
      startedAt,
      canUseForExecution: false,
    });
    process.stdout.write(
      `[mec3] ${b.basketId} verdict=${s.verdict} score=${s.persistenceScore.toFixed(2)} ` +
        `obs=${s.nObservations} lastPositive=${s.lastPositive} maxFillable≈$${s.maxFillableUsdWhilePositive}\n`,
    );
  }
  process.stdout.write(`[mec3] verdicts=${JSON.stringify(verdictTally)}\n[mec3] exit_ok\n`);
}

main().catch(err => {
  process.stderr.write(`[mec3] fatal: ${err instanceof Error ? err.message : String(err)}\n`);
  process.exitCode = 1;
});
