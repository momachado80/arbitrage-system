/**
 * Shadow Simulation Profiles — fixed virtual-capital simulation configs.
 * shadow_100 and shadow_1000 with conservative defaults.
 */

import type { LatencyProfile } from "./latencyModel";

export interface ShadowProfileConfig {
  profileId: string;
  label: string;
  startingCapital: number;
  latencyProfile: LatencyProfile;
  maxCapitalPerTrade: number;
  maxCapitalPerCluster: number;
  maxCapitalPerMarket: number;
  minConfidenceToTrade: number;
  minNetCapturableEdgeToTrade: number;
  maxHoldingTimeMs: number;
  stopLossPct: number;
  takeProfitPct: number;
  feeBuffer: number;
  impactAlpha: number;
  liquidityHaircut: number;
  enabled: boolean;
  /** Adaptive challenger overrides — only for challenger profiles */
  excludedPairKeys?: string[];
  minFillRatioToTrade?: number;
  baseProfileId?: string;
  isAdaptive?: boolean;
  /** Entry calibration: stricter capturable-edge floor (overrides minNetCapturableEdgeToTrade) */
  minCapturableEdgeToTrade?: number;
  /** Entry calibration: penalty per pairKey (decimal), subtracted from capturable edge before threshold check */
  entryPairPenalties?: Record<string, number>;
  /** Narrow challenger: só abre trade quando pair/edge/fill buckets batem exatamente. Isolado, auditável. */
  narrowChallengerTarget?: {
    pairKey: string;
    capturableEdgeBucket: "2-5%";
    fillRatioBucket: "0.25-0.5";
  };
  /** Entry capfloor challenger: só abre se capturableEdgeAtEntry >= este valor (ex: 0.03) */
  entryCapfloorMinCapturableEdge?: number;
  /** Entry degratio challenger: só abre se capturable/observed >= este valor (ex: 0.22) */
  entryDegRatioMin?: number;
}

export const SHADOW_PROFILES: ShadowProfileConfig[] = [
  {
    profileId: "shadow_100",
    label: "Shadow 500 USD (test)",
    startingCapital: 500,
    latencyProfile: "normal",
    maxCapitalPerTrade: 25,
    maxCapitalPerCluster: 50,
    maxCapitalPerMarket: 30,
    minConfidenceToTrade: 0.18,
    minNetCapturableEdgeToTrade: 0.006,
    maxHoldingTimeMs: 300_000,
    stopLossPct: 0.03,
    takeProfitPct: 0.05,
    feeBuffer: 0.002,
    impactAlpha: 1.25,
    liquidityHaircut: 0.65,
    enabled: true,
  },
  {
    profileId: "shadow_1000",
    label: "Shadow 5000 USD (test)",
    startingCapital: 5000,
    latencyProfile: "normal",
    maxCapitalPerTrade: 150,
    maxCapitalPerCluster: 400,
    maxCapitalPerMarket: 200,
    minConfidenceToTrade: 0.2,
    minNetCapturableEdgeToTrade: 0.007,
    maxHoldingTimeMs: 300_000,
    stopLossPct: 0.03,
    takeProfitPct: 0.05,
    feeBuffer: 0.002,
    impactAlpha: 1.3,
    liquidityHaircut: 0.6,
    enabled: true,
  },
  {
    profileId: "shadow_100_30s",
    label: "Shadow 500 USD (30s test)",
    startingCapital: 500,
    latencyProfile: "normal",
    maxCapitalPerTrade: 25,
    maxCapitalPerCluster: 50,
    maxCapitalPerMarket: 30,
    minConfidenceToTrade: 0.18,
    minNetCapturableEdgeToTrade: 0.006,
    maxHoldingTimeMs: 30_000,
    stopLossPct: 0.03,
    takeProfitPct: 0.05,
    feeBuffer: 0.002,
    impactAlpha: 1.25,
    liquidityHaircut: 0.65,
    enabled: true,
  },
  {
    profileId: "shadow_100_60s",
    label: "Shadow 500 USD (60s test)",
    startingCapital: 500,
    latencyProfile: "normal",
    maxCapitalPerTrade: 25,
    maxCapitalPerCluster: 50,
    maxCapitalPerMarket: 30,
    minConfidenceToTrade: 0.18,
    minNetCapturableEdgeToTrade: 0.006,
    maxHoldingTimeMs: 60_000,
    stopLossPct: 0.03,
    takeProfitPct: 0.05,
    feeBuffer: 0.002,
    impactAlpha: 1.25,
    liquidityHaircut: 0.65,
    enabled: true,
  },
  {
    profileId: "shadow_1000_30s",
    label: "Shadow 5000 USD (30s test)",
    startingCapital: 5000,
    latencyProfile: "normal",
    maxCapitalPerTrade: 150,
    maxCapitalPerCluster: 400,
    maxCapitalPerMarket: 200,
    minConfidenceToTrade: 0.2,
    minNetCapturableEdgeToTrade: 0.007,
    maxHoldingTimeMs: 30_000,
    stopLossPct: 0.03,
    takeProfitPct: 0.05,
    feeBuffer: 0.002,
    impactAlpha: 1.3,
    liquidityHaircut: 0.6,
    enabled: true,
  },
  {
    profileId: "shadow_1000_60s",
    label: "Shadow 5000 USD (60s test)",
    startingCapital: 5000,
    latencyProfile: "normal",
    maxCapitalPerTrade: 150,
    maxCapitalPerCluster: 400,
    maxCapitalPerMarket: 200,
    minConfidenceToTrade: 0.2,
    minNetCapturableEdgeToTrade: 0.007,
    maxHoldingTimeMs: 60_000,
    stopLossPct: 0.03,
    takeProfitPct: 0.05,
    feeBuffer: 0.002,
    impactAlpha: 1.3,
    liquidityHaircut: 0.6,
    enabled: true,
  },
  {
    profileId: "shadow_1000_entry_capfloor_v1",
    label: "Entry capfloor capturable>=3% (v1)",
    startingCapital: 5000,
    latencyProfile: "normal",
    maxCapitalPerTrade: 150,
    maxCapitalPerCluster: 400,
    maxCapitalPerMarket: 200,
    minConfidenceToTrade: 0.2,
    minNetCapturableEdgeToTrade: 0.007,
    maxHoldingTimeMs: 300_000,
    stopLossPct: 0.03,
    takeProfitPct: 0.05,
    feeBuffer: 0.002,
    impactAlpha: 1.3,
    liquidityHaircut: 0.6,
    enabled: true,
    entryCapfloorMinCapturableEdge: 0.03,
  },
  {
    profileId: "shadow_1000_entry_degratio_v1",
    label: "Entry degratio capturable/observed>=0.22 (v1)",
    startingCapital: 5000,
    latencyProfile: "normal",
    maxCapitalPerTrade: 150,
    maxCapitalPerCluster: 400,
    maxCapitalPerMarket: 200,
    minConfidenceToTrade: 0.2,
    minNetCapturableEdgeToTrade: 0.007,
    maxHoldingTimeMs: 300_000,
    stopLossPct: 0.03,
    takeProfitPct: 0.05,
    feeBuffer: 0.002,
    impactAlpha: 1.3,
    liquidityHaircut: 0.6,
    enabled: true,
    entryDegRatioMin: 0.22,
  },
  {
    profileId: "shadow_1000_narrow_562794_567561_edge2to5_fill25to50_v1",
    label: "Narrow 562794+567561 edge2-5% fill0.25-0.5 (v1)",
    startingCapital: 5000,
    latencyProfile: "normal",
    maxCapitalPerTrade: 150,
    maxCapitalPerCluster: 400,
    maxCapitalPerMarket: 200,
    minConfidenceToTrade: 0.2,
    minNetCapturableEdgeToTrade: 0.007,
    maxHoldingTimeMs: 300_000,
    stopLossPct: 0.03,
    takeProfitPct: 0.05,
    feeBuffer: 0.002,
    impactAlpha: 1.3,
    liquidityHaircut: 0.6,
    enabled: true,
    narrowChallengerTarget: {
      pairKey: "562794+567561",
      capturableEdgeBucket: "2-5%",
      fillRatioBucket: "0.25-0.5",
    },
  },
];

export function getEnabledProfiles(): ShadowProfileConfig[] {
  return SHADOW_PROFILES.filter((p) => p.enabled);
}

export function getProfileById(profileId: string): ShadowProfileConfig | undefined {
  return SHADOW_PROFILES.find((p) => p.profileId === profileId);
}
