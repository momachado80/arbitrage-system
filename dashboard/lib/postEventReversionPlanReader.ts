/**
 * Plan reader puro para o Post-Event Reversion Scout.
 *
 * Responsabilidade: dado o JSON do plano de catalisadores (produzido por
 * buildCatalystObservationPlan) e o snapshot atual de mercados, identificar
 * quais janelas de observação estão due no instante corrente.
 *
 * Funções puras, sem rede, sem I/O, sem .paper, sem execução, sem microcapital.
 * Extraído de scripts/runPostEventReversionScout.ts para ser testável a partir
 * de tests/ (tsconfig.tests.json só inclui lib/ + tests/).
 */

import type { NormalizedMarket } from "./polymarketClient";
import {
  isHypothesisEligibleMarket,
  HYPOTHESIS_VERSION,
  type HypothesisSport,
} from "./postEventReversionHypothesis";

export const TARGET_WINDOW_TYPES: ReadonlySet<string> = new Set([
  "PRE_EVENT_15M",
  "POST_EVENT_15M",
  "POST_EVENT_60M",
]);

export interface PlanWindowSlot {
  windowType: string;
  runAtUtc: string;
  reason?: string;
}

export interface PlanRow {
  marketId: string;
  label?: string | null;
  catalystReadinessVerdict?: string;
  observationWindows?: PlanWindowSlot[];
  /** Schema atual emitido por buildCatalystObservationPlan. */
  nextEvent?: { eventStartUtc?: string | null } | null;
  /** Schema legado/defensivo: aceito como fallback para tolerar planos antigos.
   *  buildCatalystObservationPlan não emite este campo no top-level. */
  nextEventStartUtc?: string | null;
}

export interface PlanFile {
  plan?: PlanRow[];
}

export interface DueTarget {
  marketId: string;
  question: string;
  sport: HypothesisSport;
  catalystEventStartUtc: string;
  windowType: string;
  runAtUtc: string;
  market: NormalizedMarket;
}

export function isDueWithinTolerance(
  runAtUtc: string,
  now: Date,
  toleranceMin: number,
): boolean {
  const t = new Date(runAtUtc).getTime();
  if (!Number.isFinite(t)) return false;
  const diff = Math.abs(now.getTime() - t);
  return diff <= toleranceMin * 60_000;
}

export function buildDedupeKey(
  marketId: string,
  windowType: string,
  runAtUtc: string,
): string {
  return `${marketId}|${windowType}|${runAtUtc}|${HYPOTHESIS_VERSION}`;
}

/** Lê o eventStartUtc do catalisador: prefere nested nextEvent.eventStartUtc
 *  (schema atual), aceita top-level nextEventStartUtc como fallback. */
function readCatalystStart(row: PlanRow): string | undefined {
  const nested = row.nextEvent?.eventStartUtc?.trim();
  if (nested) return nested;
  const flat = row.nextEventStartUtc?.trim();
  if (flat) return flat;
  return undefined;
}

export function collectDueTargets(
  plan: PlanFile,
  marketsById: Map<string, NormalizedMarket>,
  nowIsoStr: string,
  now: Date,
  toleranceMinutes: number,
): DueTarget[] {
  const out: DueTarget[] = [];
  for (const row of plan.plan ?? []) {
    if (row.catalystReadinessVerdict !== "HAS_NEAR_CATALYST") continue;
    const market = marketsById.get(row.marketId);
    if (!market) continue;
    const elig = isHypothesisEligibleMarket(market, nowIsoStr);
    if (!elig.eligible) continue;
    const catalystStart = readCatalystStart(row);
    if (!catalystStart) continue;
    for (const w of row.observationWindows ?? []) {
      if (!TARGET_WINDOW_TYPES.has(w.windowType)) continue;
      if (!isDueWithinTolerance(w.runAtUtc, now, toleranceMinutes)) continue;
      out.push({
        marketId: market.id,
        question: market.question,
        sport: elig.sport,
        catalystEventStartUtc: catalystStart,
        windowType: w.windowType,
        runAtUtc: w.runAtUtc,
        market,
      });
    }
  }
  return out;
}
