/**
 * Memória operacional leve: agregados por perfil de trade (sourceType|opportunityType)
 * para alimentar o factor de progresso na entrada sem ML nem persistência em disco.
 */

import type { ExitCondition, PaperOpportunityType, PaperSourceType } from "./paperTypes";

const GLOBAL_KEY = "__paperEntryProfileMemory_v1";

export type ExitProfileBucket = "no_progress_exit" | "take_profit" | "incremental_value_too_low" | "other";

type Agg = {
  noProgress: number;
  takeProfit: number;
  incrementalLow: number;
  other: number;
};

function getMap(): Map<string, Agg> {
  const g = globalThis as unknown as Record<string, Map<string, Agg>>;
  if (!g[GLOBAL_KEY]) g[GLOBAL_KEY] = new Map();
  return g[GLOBAL_KEY]!;
}

export function entryProfileKey(sourceType: PaperSourceType, opportunityType: PaperOpportunityType): string {
  return `${sourceType}|${opportunityType}`;
}

function bucketExit(c: ExitCondition): ExitProfileBucket {
  if (c === "no_progress_exit") return "no_progress_exit";
  if (c === "take_profit") return "take_profit";
  if (c === "incremental_value_too_low") return "incremental_value_too_low";
  return "other";
}

/** Incrementa contadores por perfil quando um trade fecha. */
export function recordPaperExitProfileOutcome(profileKey: string, exitCondition: ExitCondition): void {
  const m = getMap();
  let a = m.get(profileKey);
  if (!a) {
    a = { noProgress: 0, takeProfit: 0, incrementalLow: 0, other: 0 };
    m.set(profileKey, a);
  }
  const b = bucketExit(exitCondition);
  if (b === "no_progress_exit") a.noProgress += 1;
  else if (b === "take_profit") a.takeProfit += 1;
  else if (b === "incremental_value_too_low") a.incrementalLow += 1;
  else a.other += 1;
}

/** Taxa histórica [0,1] de no_progress para o perfil; 0 se ainda não há amostras suficientes. */
export function getHistoricalNoProgressRate(profileKey: string, minSamples: number): number {
  const a = getMap().get(profileKey);
  if (!a) return 0;
  const total = a.noProgress + a.takeProfit + a.incrementalLow + a.other;
  if (total < minSamples) return 0;
  return a.noProgress / total;
}

/** Taxa histórica [0,1] de take_profit para o perfil; 0 se ainda não há amostras suficientes. */
export function getHistoricalTakeProfitRate(profileKey: string, minSamples: number): number {
  const a = getMap().get(profileKey);
  if (!a) return 0;
  const total = a.noProgress + a.takeProfit + a.incrementalLow + a.other;
  if (total < minSamples) return 0;
  return a.takeProfit / total;
}

export type PaperExitProfileMemoryRow = Agg & { total: number; noProgressRate: number };

export function getPaperExitProfileMemorySnapshot(): Record<string, PaperExitProfileMemoryRow> {
  const m = getMap();
  const out: Record<string, PaperExitProfileMemoryRow> = {};
  for (const [k, a] of Array.from(m.entries())) {
    const total = a.noProgress + a.takeProfit + a.incrementalLow + a.other;
    out[k] = {
      ...a,
      total,
      noProgressRate: total > 0 ? a.noProgress / total : 0,
    };
  }
  return out;
}

function clamp01(x: number): number {
  if (!Number.isFinite(x)) return 0;
  return Math.max(0, Math.min(1, x));
}

function round6(n: number): number {
  return Math.round(n * 1e6) / 1e6;
}

/** Uma linha por perfil para API / decisão: guard global + eventual bump se histórico for desfavorável. */
export type EffectiveProgressGuardRow = {
  profileKey: string;
  globalMin: number;
  effectiveMin: number;
  adaptiveApplied: boolean;
  minSamplesRequired: number;
  totalSamples: number;
  noProgressExitCount: number;
  takeProfitCount: number;
  incrementalLowCount: number;
  otherExitCount: number;
  /** max(0, taxa_np − taxa_tp): perfis com muitos no_progress vs take_profit sobem o limiar. */
  stress: number;
};

export type AdaptiveProgressGuardOpts = {
  enableAdaptive: boolean;
  minSamples: number;
  /** Máximo a somar ao global quando stress=1 (ex.: 0,08 → de 0,12 para 0,20). */
  extraMax: number;
};

/**
 * Baseline = guard global; com amostras suficientes, eleva o mínimo efectivo se no_progress dominar sobre take_profit.
 * Nunca baixa do global (safety net).
 */
export function getEffectiveMinProgressProbabilityFactor(
  profileKey: string,
  globalMin: number,
  opts: AdaptiveProgressGuardOpts
): EffectiveProgressGuardRow {
  const empty: EffectiveProgressGuardRow = {
    profileKey,
    globalMin,
    effectiveMin: globalMin,
    adaptiveApplied: false,
    minSamplesRequired: opts.minSamples,
    totalSamples: 0,
    noProgressExitCount: 0,
    takeProfitCount: 0,
    incrementalLowCount: 0,
    otherExitCount: 0,
    stress: 0,
  };

  if (!opts.enableAdaptive) {
    return empty;
  }

  const a = getMap().get(profileKey);
  if (!a) {
    return empty;
  }

  const total = a.noProgress + a.takeProfit + a.incrementalLow + a.other;
  const base = {
    ...empty,
    totalSamples: total,
    noProgressExitCount: a.noProgress,
    takeProfitCount: a.takeProfit,
    incrementalLowCount: a.incrementalLow,
    otherExitCount: a.other,
  };

  if (total < opts.minSamples) {
    return base;
  }

  const npRate = a.noProgress / total;
  const tpRate = a.takeProfit / total;
  const stress = Math.max(0, npRate - tpRate);
  const bump = opts.extraMax * clamp01(stress);
  const effectiveMin = Math.min(0.98, globalMin + bump);

  return {
    ...base,
    effectiveMin,
    adaptiveApplied: bump > 1e-12,
    stress: round6(stress),
  };
}

/** Mapa completo para `/api/paper/system` (todos os perfis vistos em fechos). */
export function buildEffectiveProgressGuardByProfileMap(
  globalMin: number,
  opts: AdaptiveProgressGuardOpts
): Record<string, EffectiveProgressGuardRow> {
  const m = getMap();
  const out: Record<string, EffectiveProgressGuardRow> = {};
  for (const k of Array.from(m.keys())) {
    out[k] = getEffectiveMinProgressProbabilityFactor(k, globalMin, opts);
  }
  return out;
}
