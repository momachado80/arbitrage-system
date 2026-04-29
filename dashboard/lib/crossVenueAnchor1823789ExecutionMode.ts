/**
 * Modos de execução single-track 1823789 — sem novas estratégias; caps operacionais apenas.
 */

export type CrossVenueAnchor1823789ExecutionMode = "controlled_paper" | "minimal_micro_live";

const MODE_ENV = "CROSS_VENUE_ANCHOR_1823789_EXECUTION_MODE";
const HALT_ENV = "CROSS_VENUE_ANCHOR_1823789_MICRO_LIVE_HALT";
const MAX_NOTIONAL_ENV = "CROSS_VENUE_ANCHOR_1823789_MAX_NOTIONAL_USD";

export function parseCrossVenueAnchor1823789ExecutionModeFromEnv(): CrossVenueAnchor1823789ExecutionMode {
  const raw = process.env[MODE_ENV]?.trim().toLowerCase();
  if (raw === "minimal_micro_live" || raw === "minimal-micro-live") return "minimal_micro_live";
  return "controlled_paper";
}

/** `CROSS_VENUE_ANCHOR_1823789_MICRO_LIVE_HALT=1` (ou true/yes) → kill-switch. */
export function parseCrossVenueAnchor1823789MicroLiveHaltFromEnv(): boolean {
  const raw = process.env[HALT_ENV]?.trim().toLowerCase();
  return raw === "1" || raw === "true" || raw === "yes";
}

/**
 * Modo efectivo: halt força rollback a paper.
 */
export function resolveEffectiveCrossVenueAnchor1823789ExecutionMode(
  requested: CrossVenueAnchor1823789ExecutionMode,
  halt: boolean,
): CrossVenueAnchor1823789ExecutionMode {
  if (halt) return "controlled_paper";
  return requested;
}

export type MaxNotionalUsdParse =
  | { ok: true; maxNotionalUsd: number }
  | { ok: false; reason: string };

export function parseCrossVenueAnchor1823789MaxNotionalUsd(): MaxNotionalUsdParse {
  const raw = process.env[MAX_NOTIONAL_ENV]?.trim();
  if (!raw) return { ok: false, reason: `${MAX_NOTIONAL_ENV}_unset` };
  const n = Number(raw);
  if (!Number.isFinite(n) || n <= 0) return { ok: false, reason: `${MAX_NOTIONAL_ENV}_invalid` };
  return { ok: true, maxNotionalUsd: n };
}
