/**
 * MEC Partition Scan — extração compartilhada de eventos negRisk da Gamma.
 *
 * Usado pelo censo de partição (runMechanicalEdgeCensusPartition) e pelo
 * amostrador de persistência (runMecPersistenceSampler). GET-only; o parsing
 * é puro. Sem ordens, sem .paper, sem microcapital, sem execução.
 */

import { fetchJson, jsonArray, jsonNumberArray } from "./mechanicalEdgeCensusFetch";

const GAMMA_EVENTS = "https://gamma-api.polymarket.com/events";
const PAGE_LIMIT = 80;
const GAMMA_HTTP_MS = 14_000;

export const MEC_PARTITION_PAGE_LIMIT = PAGE_LIMIT;

export interface PartitionLeg {
  marketId: string;
  yesToken: string;
  yesMid: number;
}

export interface PartitionEvent {
  eventId: string;
  title: string;
  endDate: unknown;
  conversionFeeFrac: number;
  legs: PartitionLeg[];
}

function num(x: unknown): number {
  const n = typeof x === "number" ? x : parseFloat(String(x));
  return Number.isFinite(n) ? n : NaN;
}

export function isPlaceholderOutcome(name: string): boolean {
  const t = name.trim().toLowerCase();
  if (t.length < 2) return true;
  return ["tbd", "n/a", "na", "...", "—", "-"].includes(t) || /^outcome\s*\d+$/i.test(t);
}

export async function loadNegRiskEventsPage(offset: number): Promise<Record<string, unknown>[]> {
  const base = `${GAMMA_EVENTS}?active=true&closed=false&limit=${PAGE_LIMIT}&offset=${offset}&order=volume&ascending=false`;
  let body: unknown;
  try {
    body = await fetchJson(`${base}&negRisk=true`, GAMMA_HTTP_MS);
  } catch {
    body = await fetchJson(base, GAMMA_HTTP_MS);
  }
  return Array.isArray(body) ? (body as Record<string, unknown>[]) : [];
}

export function extractPartitionEvent(ev: Record<string, unknown>): PartitionEvent | null {
  if (ev.negRisk !== true) return null;
  const eventId = String(ev.id ?? "");
  const title = typeof ev.title === "string" ? ev.title : typeof ev.slug === "string" ? (ev.slug as string) : "";
  const feeBips = num(ev.negRiskFeeBips);
  const conversionFeeFrac = Number.isFinite(feeBips) && feeBips > 0 ? feeBips / 10_000 : 0;

  const marketsRaw = Array.isArray(ev.markets) ? (ev.markets as Record<string, unknown>[]) : [];
  const legs: PartitionLeg[] = [];
  for (const m of marketsRaw) {
    if (!m || m.closed === true || m.active === false || m.id == null) continue;
    const outcomes = jsonArray(m.outcomes);
    if (outcomes.length === 0 || outcomes.some(isPlaceholderOutcome)) continue;
    const tokens = jsonArray(m.clobTokenIds);
    const prices = jsonNumberArray(m.outcomePrices);
    if (tokens.length === 0 || prices.length === 0) continue;
    const yesIdx = outcomes.findIndex(o => /^yes$/i.test(o.trim()));
    const idx = yesIdx >= 0 ? yesIdx : 0;
    const yesToken = idx < tokens.length ? tokens[idx]! : tokens[0]!;
    const yesMid = idx < prices.length ? prices[idx]! : prices[0]!;
    if (!yesToken || !Number.isFinite(yesMid)) continue;
    legs.push({ marketId: String(m.id), yesToken, yesMid });
  }
  if (legs.length < 2) return null;
  return { eventId, title, endDate: ev.endDate ?? ev.end_date ?? null, conversionFeeFrac, legs };
}
