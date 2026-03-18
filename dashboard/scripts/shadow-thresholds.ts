/**
 * Configurable thresholds for shadow validation pipeline.
 * Override via env: SHADOW_MIN_CLOSED_READABLE, SHADOW_MIN_CLOSED_COMPARE, SHADOW_MIN_IMPROVEMENT_PCT
 */

function envNum(name: string, defaultVal: number): number {
  const v = process.env[name];
  if (v == null || v === "") return defaultVal;
  const n = parseFloat(v);
  return Number.isNaN(n) ? defaultVal : n;
}

export const THRESHOLDS = {
  /** Minimum closed trades for economic read to be considered valid */
  minimumClosedForReadableEconomics: envNum("SHADOW_MIN_CLOSED_READABLE", 10),

  /** Minimum closed trades for meaningful baseline comparison */
  minimumClosedForBaselineComparison: envNum("SHADOW_MIN_CLOSED_COMPARE", 20),

  /** Minimum relative improvement (e.g. 0.1 = 10% less bad) to classify as LESS_BAD_THAN_BASELINE */
  minimumRelativeImprovementToCallLessBad: envNum("SHADOW_MIN_IMPROVEMENT_PCT", 0.1),
} as const;
