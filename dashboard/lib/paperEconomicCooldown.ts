/**
 * Cooldown operacional pós-falha económica (mesmo dedupeKey): reduz reciclagem sem alterar thresholds.
 * Estado em globalThis.
 */

import type { PaperEntryEconomicRejectionReason } from "./paperEntryEconomics";

const GLOBAL_KEY = "__paperEconomicCooldown_v1";

function envBool(name: string, defaultValue: boolean): boolean {
  const raw = process.env[name]?.trim().toLowerCase();
  if (!raw) return defaultValue;
  return raw === "1" || raw === "true" || raw === "yes";
}

function envNum(name: string, defaultValue: number): number {
  const raw = process.env[name]?.trim();
  if (!raw) return defaultValue;
  const n = Number(raw);
  return Number.isFinite(n) ? n : defaultValue;
}

export type PaperEconomicCooldownPolicySnapshot = {
  enabled: boolean;
  minFailuresBeforeCooldown: number;
  cooldownMinutes: number;
  requireStrongPattern: boolean;
};

export function getEconomicCooldownPolicySnapshot(): PaperEconomicCooldownPolicySnapshot {
  return {
    enabled: envBool("PAPER_ECONOMIC_COOLDOWN_ENABLED", true),
    minFailuresBeforeCooldown: Math.max(1, Math.floor(envNum("PAPER_ECONOMIC_COOLDOWN_MIN_FAILURES", 3))),
    cooldownMinutes: Math.max(0.5, envNum("PAPER_ECONOMIC_COOLDOWN_MINUTES", 12)),
    requireStrongPattern: envBool("PAPER_ECONOMIC_COOLDOWN_REQUIRE_STRONG_PATTERN", true),
  };
}

type CooldownState = {
  cooldownUntilMs: Map<string, number>;
  /** Falhas consecutivas com o padrão alvo (reset em pass ou motivo/padrão diferente). */
  patternFailStreak: Map<string, number>;
};

function getCooldownState(): CooldownState {
  const g = globalThis as unknown as Record<string, CooldownState>;
  if (!g[GLOBAL_KEY]) {
    g[GLOBAL_KEY] = { cooldownUntilMs: new Map(), patternFailStreak: new Map() };
  }
  return g[GLOBAL_KEY]!;
}

/** Chave alinhada a `economicCandidateDedupeKey` em paperOpenDiagnostics. */
export function makeEconomicDedupeKey(profileKey: string, opportunityId: string): string {
  return `${profileKey}\u001f${opportunityId}`;
}

export function isEconomicCooldownActive(dedupeKey: string): boolean {
  const cfg = getEconomicCooldownPolicySnapshot();
  if (!cfg.enabled) return false;
  const t = getCooldownState().cooldownUntilMs.get(dedupeKey);
  if (t == null) return false;
  const now = Date.now();
  if (now >= t) {
    getCooldownState().cooldownUntilMs.delete(dedupeKey);
    return false;
  }
  return true;
}

/**
 * Após avaliação económica completa: actualiza streak e activa cooldown se aplicável.
 * Não altera decisão — só estado para o próximo ciclo.
 */
export function recordEconomicOutcomeForCooldown(
  dedupeKey: string,
  passed: boolean,
  rejectionFinal: PaperEntryEconomicRejectionReason | null,
  reasonsAll: PaperEntryEconomicRejectionReason[]
): void {
  const cfg = getEconomicCooldownPolicySnapshot();
  if (!cfg.enabled) return;
  const s = getCooldownState();
  if (passed) {
    s.patternFailStreak.delete(dedupeKey);
    return;
  }
  if (rejectionFinal !== "PROGRESS_PROBABILITY_FACTOR_BELOW_MIN") {
    s.patternFailStreak.delete(dedupeKey);
    return;
  }
  if (cfg.requireStrongPattern) {
    if (
      !reasonsAll.includes("ENTRY_ECONOMIC_SCORE_BELOW_MIN") ||
      !reasonsAll.includes("CROSS_MARKET_NET_TO_GROSS_EDGE_BELOW_MIN")
    ) {
      s.patternFailStreak.delete(dedupeKey);
      return;
    }
  }
  const n = (s.patternFailStreak.get(dedupeKey) ?? 0) + 1;
  s.patternFailStreak.set(dedupeKey, n);
  if (n >= cfg.minFailuresBeforeCooldown) {
    s.cooldownUntilMs.set(dedupeKey, Date.now() + cfg.cooldownMinutes * 60_000);
    s.patternFailStreak.delete(dedupeKey);
  }
}

export function getEconomicCooldownActiveSummary(): { activeCount: number; sampleDedupeKeys: string[] } {
  const cfg = getEconomicCooldownPolicySnapshot();
  if (!cfg.enabled) return { activeCount: 0, sampleDedupeKeys: [] };
  const now = Date.now();
  const m = getCooldownState().cooldownUntilMs;
  const active: string[] = [];
  for (const [k, until] of Array.from(m.entries())) {
    if (until > now) active.push(k);
  }
  return { activeCount: active.length, sampleDedupeKeys: active.slice(0, 12) };
}
