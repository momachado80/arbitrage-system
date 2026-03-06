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
}

export const SHADOW_PROFILES: ShadowProfileConfig[] = [
  {
    profileId: "shadow_100",
    label: "Shadow 100 USD",
    startingCapital: 100,
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
    label: "Shadow 1000 USD",
    startingCapital: 1000,
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
];

export function getEnabledProfiles(): ShadowProfileConfig[] {
  return SHADOW_PROFILES.filter((p) => p.enabled);
}

export function getProfileById(profileId: string): ShadowProfileConfig | undefined {
  return SHADOW_PROFILES.find((p) => p.profileId === profileId);
}
