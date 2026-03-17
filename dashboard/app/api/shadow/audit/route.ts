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
