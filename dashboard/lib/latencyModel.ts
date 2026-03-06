/**
 * Latency Model — models realistic delay between detection and execution.
 * Longer latency reduces capturable edge.
 */

export type LatencyProfile = "fast" | "normal" | "conservative";

export interface LatencyConfig {
  detectionDelayMs: { min: number; max: number };
  decisionDelayMs: { min: number; max: number };
  submissionDelayMs: { min: number; max: number };
  exchangeDelayMs: { min: number; max: number };
}

const PROFILES: Record<LatencyProfile, LatencyConfig> = {
  fast: {
    detectionDelayMs: { min: 50, max: 150 },
    decisionDelayMs: { min: 20, max: 80 },
    submissionDelayMs: { min: 30, max: 100 },
    exchangeDelayMs: { min: 100, max: 300 },
  },
  normal: {
    detectionDelayMs: { min: 150, max: 400 },
    decisionDelayMs: { min: 80, max: 200 },
    submissionDelayMs: { min: 100, max: 250 },
    exchangeDelayMs: { min: 200, max: 500 },
  },
  conservative: {
    detectionDelayMs: { min: 400, max: 1000 },
    decisionDelayMs: { min: 200, max: 500 },
    submissionDelayMs: { min: 250, max: 600 },
    exchangeDelayMs: { min: 400, max: 1200 },
  },
};

export interface LatencySample {
  detectionDelayMs: number;
  decisionDelayMs: number;
  submissionDelayMs: number;
  exchangeDelayMs: number;
  totalEntryLatencyMs: number;
  totalExitLatencyMs: number;
}

function sampleInRange(min: number, max: number): number {
  return min + (max - min) * (0.3 + 0.4 * Math.random());
}

export function sampleEntryLatency(profile: LatencyProfile): LatencySample {
  const cfg = PROFILES[profile] ?? PROFILES.normal;
  const detection = sampleInRange(cfg.detectionDelayMs.min, cfg.detectionDelayMs.max);
  const decision = sampleInRange(cfg.decisionDelayMs.min, cfg.decisionDelayMs.max);
  const submission = sampleInRange(cfg.submissionDelayMs.min, cfg.submissionDelayMs.max);
  const exchange = sampleInRange(cfg.exchangeDelayMs.min, cfg.exchangeDelayMs.max);
  const total = detection + decision + submission + exchange;
  return {
    detectionDelayMs: Math.round(detection),
    decisionDelayMs: Math.round(decision),
    submissionDelayMs: Math.round(submission),
    exchangeDelayMs: Math.round(exchange),
    totalEntryLatencyMs: Math.round(total),
    totalExitLatencyMs: Math.round(submission + exchange),
  };
}

export function sampleExitLatency(profile: LatencyProfile): Pick<LatencySample, "totalExitLatencyMs"> {
  const cfg = PROFILES[profile] ?? PROFILES.normal;
  const submission = sampleInRange(cfg.submissionDelayMs.min, cfg.submissionDelayMs.max);
  const exchange = sampleInRange(cfg.exchangeDelayMs.min, cfg.exchangeDelayMs.max);
  return { totalExitLatencyMs: Math.round(submission + exchange) };
}

export interface PersistenceData {
  avgActiveDurationMs?: number;
  longestRecentEpisodeMs?: number;
  edgeTrend?: "rising" | "stable" | "falling";
}

export function estimateLatencyPenalty(
  latencyMs: number,
  edgePersistenceData?: PersistenceData | null
): number {
  const basePenalty = Math.min(0.5, (latencyMs / 5000) * 0.15);
  if (!edgePersistenceData) return basePenalty;

  const avgDuration = edgePersistenceData.avgActiveDurationMs ?? 30000;
  const fragilityFactor = avgDuration < 10000 ? 1.5 : avgDuration < 30000 ? 1.2 : 1;
  const trendFactor =
    edgePersistenceData.edgeTrend === "falling" ? 1.3 : edgePersistenceData.edgeTrend === "rising" ? 0.9 : 1;

  return Math.min(0.6, basePenalty * fragilityFactor * trendFactor);
}
