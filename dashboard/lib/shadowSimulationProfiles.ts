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
  /** Structural challenger: pair set + fill bucket + edge bucket opcional */
  structuralChallengerTarget?: {
    pairKeys: readonly string[];
    fillRatioBucket: "0.1-0.25";
    capturableEdgeBucket: ">5%" | null;
  };
  /** Structural risk-managed: pair set + fill + entry capfloor + degratio + adaptive sizing */
  structuralRiskManagedTarget?: {
    pairKeys: readonly string[];
    fillRatioBucket: "0.1-0.25" | "0.1-0.5";
    capfloor: number;
    degRatioMin: number;
  };
  /** Exit kill: saída adaptativa mais agressiva. Requer structuralRiskManagedTarget ( mesma entrada ). */
  exitKillTarget?: {
    monitoringWindowMs: number;
    killCapturableDecayFraction: number;
    killNetEdgeFloor: number;
    killAbsentCycles: number;
    killObservedEdgeDecayFraction: number;
  };
  /** Late exit: saída por não reversão ou estagnação tardia. Hipótese diferente de kill precoce. */
  lateExitTarget?: {
    minObservationMs: number;
    reversionMinFraction: number;
    stagnantEdgeFloor: number;
    stagnantCycles: number;
    netEdgeProlongedFloor: number;
    absentCyclesInLatePhase: number;
  };
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
  {
    profileId: "shadow_1000_structural_narrow_gt5_fill10to25_v1",
    label: "Structural pair×fill0.1-0.25×edge>5% (v1)",
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
    structuralChallengerTarget: {
      pairKeys: [
        "540817+565065",
        "540817+562187",
        "540817+573647",
        "540817+540818",
        "556108+567561",
        "556108+562187",
        "540818+556108",
      ],
      fillRatioBucket: "0.1-0.25",
      capturableEdgeBucket: ">5%",
    },
  },
  {
    profileId: "shadow_1000_structural_narrow_fill10to25_v1",
    label: "Structural pair×fill0.1-0.25 (v1)",
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
    structuralChallengerTarget: {
      pairKeys: [
        "540817+565065",
        "540817+562187",
        "540817+573647",
        "540817+540818",
        "556108+567561",
        "556108+562187",
        "540818+556108",
      ],
      fillRatioBucket: "0.1-0.25",
      capturableEdgeBucket: null,
    },
  },
  {
    profileId: "shadow_1000_structural_riskmanaged_v1",
    label: "Structural risk-managed pair×fill×capfloor×degratio (v1)",
    startingCapital: 5000,
    latencyProfile: "normal",
    maxCapitalPerTrade: 150,
    maxCapitalPerCluster: 400,
    maxCapitalPerMarket: 200,
    minConfidenceToTrade: 0.2,
    minNetCapturableEdgeToTrade: 0.045,
    maxHoldingTimeMs: 300_000,
    stopLossPct: 0.03,
    takeProfitPct: 0.05,
    feeBuffer: 0.002,
    impactAlpha: 1.3,
    liquidityHaircut: 0.6,
    enabled: true,
    structuralRiskManagedTarget: {
      pairKeys: [
        "540817+565065",
        "540817+562187",
        "540817+573647",
        "540817+540818",
        "556108+567561",
        "556108+562187",
        "540818+556108",
      ],
      fillRatioBucket: "0.1-0.25",
      capfloor: 0.045,
      degRatioMin: 0.24,
    },
  },
  {
    profileId: "shadow_1000_structural_fillrelax_v1",
    label: "Structural fill-relax: pair×fill0.1-0.5×capfloor×degratio (v1)",
    startingCapital: 5000,
    latencyProfile: "normal",
    maxCapitalPerTrade: 150,
    maxCapitalPerCluster: 400,
    maxCapitalPerMarket: 200,
    minConfidenceToTrade: 0.2,
    minNetCapturableEdgeToTrade: 0.045,
    maxHoldingTimeMs: 300_000,
    stopLossPct: 0.03,
    takeProfitPct: 0.05,
    feeBuffer: 0.002,
    impactAlpha: 1.3,
    liquidityHaircut: 0.6,
    enabled: true,
    structuralRiskManagedTarget: {
      pairKeys: [
        "540817+565065",
        "540817+562187",
        "540817+573647",
        "540817+540818",
        "556108+567561",
        "556108+562187",
        "540818+556108",
      ],
      fillRatioBucket: "0.1-0.5",
      capfloor: 0.045,
      degRatioMin: 0.24,
    },
  },
  {
    profileId: "shadow_1000_structural_fillrelax_capfloorrelax_v1",
    label: "Structural fill+capfloor-relax: pair×fill0.1-0.5×capfloor3%×degratio (v1)",
    startingCapital: 5000,
    latencyProfile: "normal",
    maxCapitalPerTrade: 150,
    maxCapitalPerCluster: 400,
    maxCapitalPerMarket: 200,
    minConfidenceToTrade: 0.2,
    minNetCapturableEdgeToTrade: 0.045,
    maxHoldingTimeMs: 300_000,
    stopLossPct: 0.03,
    takeProfitPct: 0.05,
    feeBuffer: 0.002,
    impactAlpha: 1.3,
    liquidityHaircut: 0.6,
    enabled: true,
    structuralRiskManagedTarget: {
      pairKeys: [
        "540817+565065",
        "540817+562187",
        "540817+573647",
        "540817+540818",
        "556108+567561",
        "556108+562187",
        "540818+556108",
      ],
      fillRatioBucket: "0.1-0.5",
      capfloor: 0.03,
      degRatioMin: 0.24,
    },
  },
  {
    profileId: "shadow_1000_structural_exitkill_v1",
    label: "Structural exit-kill: mesma entrada, saída mais agressiva (v1)",
    startingCapital: 5000,
    latencyProfile: "normal",
    maxCapitalPerTrade: 150,
    maxCapitalPerCluster: 400,
    maxCapitalPerMarket: 200,
    minConfidenceToTrade: 0.2,
    minNetCapturableEdgeToTrade: 0.045,
    maxHoldingTimeMs: 300_000,
    stopLossPct: 0.03,
    takeProfitPct: 0.05,
    feeBuffer: 0.002,
    impactAlpha: 1.3,
    liquidityHaircut: 0.6,
    enabled: true,
    structuralRiskManagedTarget: {
      pairKeys: [
        "540817+565065",
        "540817+562187",
        "540817+573647",
        "540817+540818",
        "556108+567561",
        "556108+562187",
        "540818+556108",
      ],
      fillRatioBucket: "0.1-0.25",
      capfloor: 0.045,
      degRatioMin: 0.24,
    },
    exitKillTarget: {
      monitoringWindowMs: 90_000,
      killCapturableDecayFraction: 0.5,
      killNetEdgeFloor: 0.02,
      killAbsentCycles: 2,
      killObservedEdgeDecayFraction: 0.5,
    },
  },
  {
    profileId: "shadow_1000_structural_exitkill_window180_v1",
    label: "Structural exit-kill window 180s: mesma entrada, janela ampliada (v1)",
    startingCapital: 5000,
    latencyProfile: "normal",
    maxCapitalPerTrade: 150,
    maxCapitalPerCluster: 400,
    maxCapitalPerMarket: 200,
    minConfidenceToTrade: 0.2,
    minNetCapturableEdgeToTrade: 0.045,
    maxHoldingTimeMs: 300_000,
    stopLossPct: 0.03,
    takeProfitPct: 0.05,
    feeBuffer: 0.002,
    impactAlpha: 1.3,
    liquidityHaircut: 0.6,
    enabled: true,
    structuralRiskManagedTarget: {
      pairKeys: [
        "540817+565065",
        "540817+562187",
        "540817+573647",
        "540817+540818",
        "556108+567561",
        "556108+562187",
        "540818+556108",
      ],
      fillRatioBucket: "0.1-0.25",
      capfloor: 0.045,
      degRatioMin: 0.24,
    },
    exitKillTarget: {
      monitoringWindowMs: 180_000,
      killCapturableDecayFraction: 0.5,
      killNetEdgeFloor: 0.02,
      killAbsentCycles: 2,
      killObservedEdgeDecayFraction: 0.5,
    },
  },
  {
    profileId: "shadow_1000_structural_lateexit_nonreversion_v1",
    label: "Structural late exit: não reversão / estagnação tardia (v1)",
    startingCapital: 5000,
    latencyProfile: "normal",
    maxCapitalPerTrade: 150,
    maxCapitalPerCluster: 400,
    maxCapitalPerMarket: 200,
    minConfidenceToTrade: 0.2,
    minNetCapturableEdgeToTrade: 0.045,
    maxHoldingTimeMs: 300_000,
    stopLossPct: 0.03,
    takeProfitPct: 0.05,
    feeBuffer: 0.002,
    impactAlpha: 1.3,
    liquidityHaircut: 0.6,
    enabled: true,
    structuralRiskManagedTarget: {
      pairKeys: [
        "540817+565065",
        "540817+562187",
        "540817+573647",
        "540817+540818",
        "556108+567561",
        "556108+562187",
        "540818+556108",
      ],
      fillRatioBucket: "0.1-0.25",
      capfloor: 0.045,
      degRatioMin: 0.24,
    },
    lateExitTarget: {
      minObservationMs: 90_000,
      reversionMinFraction: 0.5,
      stagnantEdgeFloor: 0.03,
      stagnantCycles: 3,
      netEdgeProlongedFloor: 0.02,
      absentCyclesInLatePhase: 2,
    },
  },
  {
    profileId: "shadow_1000_structural_lateexit_tighter_v1",
    label: "Structural late exit tighter: thresholds apertados (v1)",
    startingCapital: 5000,
    latencyProfile: "normal",
    maxCapitalPerTrade: 150,
    maxCapitalPerCluster: 400,
    maxCapitalPerMarket: 200,
    minConfidenceToTrade: 0.2,
    minNetCapturableEdgeToTrade: 0.045,
    maxHoldingTimeMs: 300_000,
    stopLossPct: 0.03,
    takeProfitPct: 0.05,
    feeBuffer: 0.002,
    impactAlpha: 1.3,
    liquidityHaircut: 0.6,
    enabled: true,
    structuralRiskManagedTarget: {
      pairKeys: [
        "540817+565065",
        "540817+562187",
        "540817+573647",
        "540817+540818",
        "556108+567561",
        "556108+562187",
        "540818+556108",
      ],
      fillRatioBucket: "0.1-0.25",
      capfloor: 0.045,
      degRatioMin: 0.24,
    },
    lateExitTarget: {
      minObservationMs: 90_000,
      reversionMinFraction: 0.5,
      stagnantEdgeFloor: 0.045,
      stagnantCycles: 3,
      netEdgeProlongedFloor: 0.03,
      absentCyclesInLatePhase: 2,
    },
  },
];

export function getEnabledProfiles(): ShadowProfileConfig[] {
  return SHADOW_PROFILES.filter((p) => p.enabled);
}

export function getProfileById(profileId: string): ShadowProfileConfig | undefined {
  return SHADOW_PROFILES.find((p) => p.profileId === profileId);
}
