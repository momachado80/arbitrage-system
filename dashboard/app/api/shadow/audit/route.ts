/**
 * Shadow Closed Trade Audit API — diagnostic endpoint only.
 * Returns full audit of closed shadow trades. No business logic changes.
 * Inclui operationalTruth: respostas causais para decisão operacional.
 */

import { NextResponse } from "next/server";
import { ensureShadowSimulation, getShadowSystemStatus, getProfileConfig, getProfilesForExecution } from "@/lib/shadowSimulationService";
import { getAllShadowProfiles, getRejectionCountsByProfile, getPersistenceStatus, getRehydratedTradeIds } from "@/lib/shadowSimulationStore";
import {
  computeClosedTradeAudit,
  buildClosedTradeAuditEntries,
  computeLocalViabilitySegments,
} from "@/lib/shadowClosedTradeAudit";
import { runStructuralResearchV2 } from "@/lib/structuralResearchV2";
import { runStructuralPersistenceValidation } from "@/lib/structuralPersistenceValidation";
import { getSelectionDiagnostics } from "@/lib/shadowSelectionDiagnostics";
import { getFillGuardDiagnostics } from "@/lib/fillGuardDiagnostics";
import { getEntryThresholdCausalDiagnostics } from "@/lib/entryThresholdCausalDiagnostics";
import { getEntryThresholdFlowDiagnostics } from "@/lib/entryThresholdFlowDiagnostics";
import { getShadowRuntimeDiagnostics } from "@/lib/shadowRuntimeDiagnostics";
import { getMarketSourceDiagnostics } from "@/lib/marketSourceDiagnostics";
import { getProfileById } from "@/lib/shadowSimulationProfiles";
import { getServiceStats } from "@/lib/marketDataService";
import { getGraphScanStats } from "@/lib/graphScanService";
import { getPipelineDiagnostics } from "@/lib/shadowPipelineDiagnostics";
import {
  getNarrowChallengerDiagnostics,
  getNarrowChallengerComparison,
} from "@/lib/narrowChallengerDiagnostics";
import {
  getStructuralChallengerDiagnostics,
  getStructuralChallengerComparison,
} from "@/lib/structuralChallengerDiagnostics";
import {
  getStructuralRiskManagedDiagnostics,
  getStructuralRiskManagedComparison,
  STRUCTURAL_RISK_MANAGED_PROFILE_ID,
} from "@/lib/structuralRiskManagedDiagnostics";
import {
  getStructuralExitKillDiagnostics,
  getStructuralExitKillComparison,
  getExitKillCausalAudit,
  STRUCTURAL_EXIT_KILL_PROFILE_ID,
} from "@/lib/structuralExitKillDiagnostics";
import {
  getStructuralExitKillWindow180Diagnostics,
  getStructuralExitKillWindow180Comparison,
  getExitKillWindow180CausalAudit,
  STRUCTURAL_EXIT_KILL_WINDOW180_PROFILE_ID,
} from "@/lib/structuralExitKillWindow180Diagnostics";
import {
  getStructuralLateExitDiagnostics,
  getStructuralLateExitComparison,
  STRUCTURAL_LATE_EXIT_PROFILE_ID,
} from "@/lib/structuralLateExitDiagnostics";
import {
  getStructuralLateExitTighterDiagnostics,
  getStructuralLateExitTighterComparison,
  STRUCTURAL_LATE_EXIT_TIGHTER_PROFILE_ID,
} from "@/lib/structuralLateExitTighterDiagnostics";
import { getStructuralLateExitCausalAudit } from "@/lib/structuralLateExitCausalAudit";
import { getExitKillComparativeProximityAudit } from "@/lib/exitKillComparativeProximityAudit";
import {
  getEntryChallengerDiagnostics,
  getEntryChallengerComparisonDiagnostics,
  getFilteredOpportunityCounterfactual,
  getEntryChallengerMetricsSummary,
  BASELINE_PROFILE_ID,
} from "@/lib/entryChallengerDiagnostics";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    ensureShadowSimulation();
    getProfilesForExecution(); // Materialize enabled challengers before audit (ensureProfileState for challengers)
    const profiles = getAllShadowProfiles();
    const audit = computeClosedTradeAudit(profiles, getProfileConfig);
    const status = getShadowSystemStatus();
    const rejectionCountsByProfile = getRejectionCountsByProfile();
    const marketStats = getServiceStats();
    const graphStats = getGraphScanStats();
    const maxHoldingTimeMsByProfile: Record<string, number> = {};
    for (const p of profiles) {
      const cfg = getProfileById(p.profileId) ?? getProfileConfig(p.profileId);
      if (cfg) maxHoldingTimeMsByProfile[p.profileId] = cfg.maxHoldingTimeMs;
    }
    const persistenceStatus = getPersistenceStatus();
    const pipelineDiagnostics = getPipelineDiagnostics();
    const newClosedThisRun =
      persistenceStatus.inMemoryClosedTradesCount - persistenceStatus.persistedClosedTradesCount;
    const selectionDiagnostics = getSelectionDiagnostics(
      profiles.map((p) => ({
        profileId: p.profileId,
        closedTrades: p.closedTrades,
        activeTrades: p.activeTrades,
      })),
      rejectionCountsByProfile
    );
    const fillGuardDiagnostics = getFillGuardDiagnostics(profiles);
    const entryThresholdCausalDiagnostics = getEntryThresholdCausalDiagnostics((pid) => {
      const cfg = getProfileById(pid) ?? getProfileConfig(pid);
      return cfg as { minNetCapturableEdgeToTrade?: number; minCapturableEdgeToTrade?: number } | undefined;
    });
    const entryThresholdFlowDiagnostics = getEntryThresholdFlowDiagnostics(profiles.map((p) => p.profileId));
    const shadowRuntimeDiagnostics = getShadowRuntimeDiagnostics(
      {
        ...marketStats,
        bootstrapPhase: marketStats.bootstrapPhase,
        refreshStuckMs: marketStats.refreshStuckMs,
      },
      { registered: true, intervalMs: 10_000 }
    );
    const marketSourceDiagnostics = getMarketSourceDiagnostics();
    const effectiveEntryThresholdByProfile: Record<string, number> = {};
    for (const p of profiles) {
      const cfg = getProfileById(p.profileId) ?? getProfileConfig(p.profileId);
      const v = cfg?.minCapturableEdgeToTrade ?? cfg?.minNetCapturableEdgeToTrade;
      if (typeof v === "number") effectiveEntryThresholdByProfile[p.profileId] = v;
    }
    const rt = shadowRuntimeDiagnostics;
    const ms = marketSourceDiagnostics;
    const ops = rt.bootstrapOperationalTroubleshooting;

    const operationalTruth = {
      /** 1. Boot da aplicação terminou? */
      bootComplete: rt.serviceBootCompleted,
      /** 2-5. Bootstrap de mercado: começou, terminou, estado */
      marketBootstrapStarted: rt.marketBootstrapAttempted,
      marketBootstrapCompleted: rt.marketBootstrapCompleted,
      marketBootstrapFailed: rt.marketBootstrapFailed,
      marketBootstrapStatus: rt.marketBootstrapStatus,
      marketBootstrapErrorMessage: rt.marketBootstrapErrorMessage,
      /** 6-7. Refresh: chamado, terminou */
      refreshCalled: rt.marketRefreshAttemptedCount > 0,
      refreshCompleted: !rt.marketRefreshPending,
      refreshCoherent: rt.marketRefreshAttemptedCount === rt.marketRefreshSuccessCount + rt.marketRefreshFailureCount,
      /** 8-10. Fetches standard, graph, merge */
      standardOpportunitiesCount: ms.standardMarketsCount,
      graphOpportunitiesCount: ms.graphMarketsCount,
      mergedOpportunitiesCount: ms.mergedMarketsCount,
      /** 11-12. Loop shadow: iniciou, gate se bloqueado */
      shadowLoopStarted: rt.shadowLoopStarted,
      shadowLoopBlockReason: rt.loopBlockReason,
      /** 13-14. Oportunidades e perfis em ciclo novo */
      opportunitiesReachingProfiles: rt.shadowLoopHeartbeatCount > 0 ? status.opportunitiesSeenLastCycle : 0,
      evaluateOpportunityCallCount: pipelineDiagnostics.totalEvaluateCalls,
      cycleCompletedCount: entryThresholdFlowDiagnostics.cycleLevel?.cycleCompletedCount ?? 0,
      /** 15. Histórico reidratado vs produção nova */
      rehydratedClosedTradesCount: persistenceStatus.persistedClosedTradesCount,
      newClosedThisRun,
      inMemoryClosedTradesCount: persistenceStatus.inMemoryClosedTradesCount,
      rehydratedAt: persistenceStatus.rehydratedAt,
      /** Ambiente operacional */
      instrumentationRanAt: rt.instrumentationRanAt,
      schedulerScheduledAt: rt.schedulerScheduledAt,
      /** Conclusão causal */
      environmentValidForEconomicInference:
        rt.marketBootstrapCompleted &&
        rt.shadowLoopStarted &&
        rt.shadowLoopHeartbeatCount > 0 &&
        (ms.standardMarketsCount > 0 || ms.graphMarketsCount > 0),
      /** Troubleshooting operacional */
      bootstrapPhase: ops?.bootstrapPhase,
      refreshStuckMs: ops?.refreshStuckMs,
      lockStuck: ops?.lockStuck,
      lastCompletedStep: ops?.lastCompletedStep,
      lastFailedStep: ops?.lastFailedStep,
      lastStepStartedAt: ops?.lastStepStartedAt,
      lastStepCompletedAt: ops?.lastStepCompletedAt,
      lastRefreshErrorMessage: ops?.lastRefreshErrorMessage,
      refreshAttemptDurationMs: ops?.refreshAttemptDurationMs,
      bootstrapAttemptDurationMs: ops?.bootstrapAttemptDurationMs,
      timeoutOperationalOccurred: ops?.timeoutOperationalOccurred,
    };

    const allAuditEntries = buildClosedTradeAuditEntries(profiles, (pid) => {
      const cfg = getProfileById(pid) ?? getProfileConfig(pid);
      return cfg as { maxHoldingTimeMs?: number } | undefined;
    });
    const localViabilityPrep = computeLocalViabilitySegments(
      allAuditEntries,
      getRehydratedTradeIds()
    );
    const structuralResearchV2 = runStructuralResearchV2(
      allAuditEntries,
      getRehydratedTradeIds()
    );
    const structuralPersistenceValidation = runStructuralPersistenceValidation(
      allAuditEntries,
      getRehydratedTradeIds()
    );
    const narrowChallengerDiagnostics = getNarrowChallengerDiagnostics(
      profiles,
      allAuditEntries,
      getRehydratedTradeIds()
    );
    const narrowChallengerComparison = getNarrowChallengerComparison(profiles, narrowChallengerDiagnostics);

    const structuralProfileConfigs = [
      {
        profileId: "shadow_1000_structural_narrow_gt5_fill10to25_v1",
        targetPairKeys: [
          "540817+565065",
          "540817+562187",
          "540817+573647",
          "540817+540818",
          "556108+567561",
          "556108+562187",
          "540818+556108",
        ],
        targetFillRatioBucket: "0.1-0.25",
        targetCapturableEdgeBucket: ">5%" as const,
      },
      {
        profileId: "shadow_1000_structural_narrow_fill10to25_v1",
        targetPairKeys: [
          "540817+565065",
          "540817+562187",
          "540817+573647",
          "540817+540818",
          "556108+567561",
          "556108+562187",
          "540818+556108",
        ],
        targetFillRatioBucket: "0.1-0.25",
        targetCapturableEdgeBucket: null,
      },
    ];
    const structuralChallengerDiagnostics: Record<string, import("@/lib/structuralChallengerDiagnostics").StructuralChallengerDiagnosticsBlock> = {};
    const structuralChallengerComparison: Record<string, import("@/lib/structuralChallengerDiagnostics").StructuralChallengerComparisonBlock> = {};
    for (const cfg of structuralProfileConfigs) {
      structuralChallengerDiagnostics[cfg.profileId] = getStructuralChallengerDiagnostics(
        profiles,
        allAuditEntries,
        cfg
      );
      structuralChallengerComparison[cfg.profileId] = getStructuralChallengerComparison(
        profiles,
        structuralChallengerDiagnostics[cfg.profileId]
      );
    }

    const profilesForEntry = profiles.map((p) => ({
      profileId: p.profileId,
      closedTrades: p.closedTrades,
      activeTrades: p.activeTrades,
    }));
    const entryChallengerDiagnostics = getEntryChallengerDiagnostics(
      profilesForEntry,
      rejectionCountsByProfile
    );
    const entryChallengerComparisonDiagnostics = getEntryChallengerComparisonDiagnostics();
    const baselineProfile = profiles.find((p) => p.profileId === BASELINE_PROFILE_ID);
    const baselineClosedForCounterfactual =
      baselineProfile?.closedTrades?.filter((t) => t.status === "closed" && t.closedAt) ?? [];
    const filteredOpportunityCounterfactual = getFilteredOpportunityCounterfactual(
      baselineClosedForCounterfactual.map((t) => ({
        opportunityId: t.opportunityId,
        realizedPnL: t.realizedPnL ?? 0,
        pairKey: t.pairKey,
        capfloorFilteredSameCycle: t.capfloorFilteredSameCycle,
        degratioFilteredSameCycle: t.degratioFilteredSameCycle,
      }))
    );
    const baselineTotalPnL = baselineClosedForCounterfactual.reduce(
      (s, t) => s + (t.realizedPnL ?? 0),
      0
    );
    const entryChallengerMetricsSummary = getEntryChallengerMetricsSummary(
      filteredOpportunityCounterfactual,
      baselineTotalPnL,
      entryChallengerDiagnostics
    );

    const structuralRiskManagedProfile = profiles.find((p) => p.profileId === STRUCTURAL_RISK_MANAGED_PROFILE_ID);
    const structuralRiskManagedDiagnostics = getStructuralRiskManagedDiagnostics(
      structuralRiskManagedProfile,
      allAuditEntries,
      rejectionCountsByProfile
    );
    const structuralRiskManagedComparison: Record<
      string,
      import("@/lib/structuralRiskManagedDiagnostics").StructuralRiskManagedComparisonBlock
    > = {};
    for (const compareId of ["shadow_1000", "shadow_1000_adapt_captrade_exitrefine_v1"]) {
      structuralRiskManagedComparison[compareId] = getStructuralRiskManagedComparison(
        profiles,
        structuralRiskManagedDiagnostics,
        compareId
      );
    }

    const structuralExitKillProfile = profiles.find((p) => p.profileId === STRUCTURAL_EXIT_KILL_PROFILE_ID);
    const structuralExitKillDiagnostics = getStructuralExitKillDiagnostics(
      structuralExitKillProfile,
      allAuditEntries,
      rejectionCountsByProfile
    );
    const structuralExitKillCausalAudit = getExitKillCausalAudit(structuralExitKillProfile);
    const structuralExitKillComparison: Record<
      string,
      import("@/lib/structuralExitKillDiagnostics").StructuralExitKillComparisonBlock
    > = {};
    for (const compareId of [
      "shadow_1000",
      "shadow_1000_adapt_captrade_exitrefine_v1",
      "shadow_1000_structural_riskmanaged_v1",
    ]) {
      structuralExitKillComparison[compareId] = getStructuralExitKillComparison(
        profiles,
        structuralExitKillDiagnostics,
        compareId
      );
    }

    const structuralExitKillWindow180Profile = profiles.find((p) => p.profileId === STRUCTURAL_EXIT_KILL_WINDOW180_PROFILE_ID);
    const structuralExitKillWindow180Diagnostics = getStructuralExitKillWindow180Diagnostics(
      structuralExitKillWindow180Profile,
      allAuditEntries,
      rejectionCountsByProfile
    );
    const structuralExitKillWindow180CausalAudit = getExitKillWindow180CausalAudit(structuralExitKillWindow180Profile);
    const structuralExitKillWindow180Comparison: Record<
      string,
      import("@/lib/structuralExitKillDiagnostics").StructuralExitKillComparisonBlock
    > = {};
    for (const compareId of [
      "shadow_1000",
      "shadow_1000_structural_riskmanaged_v1",
      "shadow_1000_structural_exitkill_v1",
    ]) {
      structuralExitKillWindow180Comparison[compareId] = getStructuralExitKillWindow180Comparison(
        profiles,
        structuralExitKillWindow180Diagnostics,
        compareId
      );
    }

    const structuralLateExitProfile = profiles.find((p) => p.profileId === STRUCTURAL_LATE_EXIT_PROFILE_ID);
    const structuralLateExitDiagnostics = getStructuralLateExitDiagnostics(
      structuralLateExitProfile,
      allAuditEntries,
      rejectionCountsByProfile
    );
    const structuralLateExitCausalAudit = getStructuralLateExitCausalAudit(structuralLateExitProfile);
    const structuralLateExitComparison: Record<
      string,
      import("@/lib/structuralLateExitDiagnostics").StructuralLateExitComparisonBlock
    > = {};
    for (const compareId of [
      "shadow_1000",
      "shadow_1000_structural_riskmanaged_v1",
      "shadow_1000_structural_exitkill_v1",
      "shadow_1000_structural_exitkill_window180_v1",
    ]) {
      structuralLateExitComparison[compareId] = getStructuralLateExitComparison(
        profiles,
        structuralLateExitDiagnostics,
        compareId
      );
    }

    const structuralLateExitTighterProfile = profiles.find((p) => p.profileId === STRUCTURAL_LATE_EXIT_TIGHTER_PROFILE_ID);
    const structuralLateExitTighterDiagnostics = getStructuralLateExitTighterDiagnostics(
      structuralLateExitTighterProfile,
      allAuditEntries,
      rejectionCountsByProfile
    );
    const structuralLateExitTighterCausalAudit = getStructuralLateExitCausalAudit(
      structuralLateExitTighterProfile,
      {
        profileId: STRUCTURAL_LATE_EXIT_TIGHTER_PROFILE_ID,
        stagnantEdgeFloor: 0.045,
        netEdgeProlongedFloor: 0.03,
      }
    );
    const structuralLateExitTighterComparison: Record<
      string,
      import("@/lib/structuralLateExitTighterDiagnostics").StructuralLateExitTighterComparisonBlock
    > = {};
    for (const compareId of [
      "shadow_1000",
      "shadow_1000_structural_riskmanaged_v1",
      "shadow_1000_structural_exitkill_v1",
      "shadow_1000_structural_exitkill_window180_v1",
      "shadow_1000_structural_lateexit_nonreversion_v1",
    ]) {
      structuralLateExitTighterComparison[compareId] = getStructuralLateExitTighterComparison(
        profiles,
        structuralLateExitTighterDiagnostics,
        compareId
      );
    }

    return NextResponse.json({
      ...audit,
      maxHoldingTimeMsByProfile,
      opportunitiesSeenLastCycle: status.opportunitiesSeenLastCycle,
      rejectionCountsByProfile,
      selectionDiagnostics,
      fillGuardDiagnostics,
      entryThresholdCausalDiagnostics,
      entryThresholdFlowDiagnostics,
      shadowRuntimeDiagnostics,
      marketSourceDiagnostics,
      effectiveEntryThresholdByProfile,
      operationalTruth,
      localViabilityPrep,
      structuralResearchV2,
      structuralPersistenceValidation,
      narrowChallengerDiagnostics,
      narrowChallengerComparison,
      structuralChallengerDiagnostics,
      structuralChallengerComparison,
      structuralRiskManagedDiagnostics,
      structuralRiskManagedComparison,
      structuralExitKillDiagnostics,
      structuralExitKillComparison,
      structuralExitKillCausalAudit,
      structuralExitKillWindow180Diagnostics,
      structuralExitKillWindow180Comparison,
      structuralExitKillWindow180CausalAudit,
      structuralLateExitDiagnostics,
      structuralLateExitComparison,
      structuralLateExitCausalAudit,
      structuralLateExitTighterDiagnostics,
      structuralLateExitTighterComparison,
      structuralLateExitTighterCausalAudit,
      exitKillComparativeProximityAudit: getExitKillComparativeProximityAudit(profiles),
      entryChallengerDiagnostics,
      entryChallengerComparisonDiagnostics,
      filteredOpportunityCounterfactual,
      entryChallengerMetricsSummary,
      persistence: {
        ...persistenceStatus,
      },
      upstreamDiagnostics: {
        marketsTracked: marketStats.marketsTracked,
        marketLastError: marketStats.lastError ?? null,
        graphOpportunitiesCount: graphStats.opportunitiesCount,
      },
    });
  } catch (err) {
    console.error("[API /shadow/audit]", err);
    return NextResponse.json(
      { error: "Audit failed", timestamp: new Date().toISOString() },
      { status: 500 }
    );
  }
}
