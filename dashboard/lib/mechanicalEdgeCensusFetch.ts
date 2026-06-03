/**
 * Mechanical Edge Census — helpers de rede e ledger (compartilhados entre runners).
 *
 * Apenas leituras HTTP (GET Gamma + GET CLOB book) e escrita append-only no JSONL.
 * Sem ordens, sem .paper, sem microcapital, sem execução. Mesmo precedente de
 * `clobMicrostructure.ts` (fetch público em lib/).
 */

import fs from "fs";
import path from "path";

import type { BookLevel } from "./mechanicalEdgeCensusBook";

export const CLOB_BASE = (
  process.env.POLYMARKET_CLOB_HOST || "https://clob.polymarket.com"
).replace(/\/$/, "");

export interface RawBook {
  bids: BookLevel[];
  asks: BookLevel[];
}

function num(x: unknown): number {
  const n = typeof x === "number" ? x : parseFloat(String(x));
  return Number.isFinite(n) ? n : NaN;
}

export async function fetchJson(url: string, ms: number): Promise<unknown> {
  const res = await fetch(url, {
    signal: AbortSignal.timeout(ms),
    headers: { Accept: "application/json" },
  });
  if (!res.ok) throw new Error(`http_${res.status}`);
  return (await res.json()) as unknown;
}

export async function fetchRawBook(tokenId: string, timeoutMs = 6_000): Promise<RawBook | null> {
  if (!tokenId) return null;
  try {
    const raw = (await fetchJson(
      `${CLOB_BASE}/book?token_id=${encodeURIComponent(tokenId)}`,
      timeoutMs,
    )) as {
      bids?: Array<{ price?: string; size?: string }>;
      asks?: Array<{ price?: string; size?: string }>;
    };
    const toLevels = (arr?: Array<{ price?: string; size?: string }>): BookLevel[] =>
      (Array.isArray(arr) ? arr : [])
        .map(l => ({ price: num(l.price), size: num(l.size) }))
        .filter(l => Number.isFinite(l.price) && Number.isFinite(l.size));
    return { bids: toLevels(raw.bids), asks: toLevels(raw.asks) };
  } catch {
    return null;
  }
}

/** Guard: o ledger jamais entra em .paper. */
export function assertSafeLedgerPath(p: string): void {
  const norm = path.resolve(p);
  if (norm.includes(`${path.sep}.paper${path.sep}`) || norm.endsWith(`${path.sep}.paper`)) {
    throw new Error("ledger_path_blocked:.paper");
  }
}

export function appendMecLedger(ledgerPath: string, entry: Record<string, unknown>): void {
  const dir = path.dirname(path.resolve(ledgerPath));
  fs.mkdirSync(dir, { recursive: true });
  fs.appendFileSync(path.resolve(ledgerPath), `${JSON.stringify(entry)}\n`, "utf8");
}

export function jsonArray(value: unknown): string[] {
  if (Array.isArray(value)) return value.map(v => String(v)).filter(Boolean);
  if (typeof value === "string" && value.trim()) {
    try {
      const p = JSON.parse(value) as unknown;
      return Array.isArray(p) ? p.map(v => String(v)).filter(Boolean) : [];
    } catch {
      return [];
    }
  }
  return [];
}

export function jsonNumberArray(value: unknown): number[] {
  return jsonArray(value)
    .map(s => num(s))
    .filter(n => Number.isFinite(n));
}
