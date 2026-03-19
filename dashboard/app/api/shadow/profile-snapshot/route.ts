/**
 * Profile Atomic Snapshot API — snapshot atômico por profile.
 * Single read, single payload. Query ?profileId=... para um profile específico.
 */

import { NextRequest, NextResponse } from "next/server";
import {
  ensureShadowSimulation,
  getProfileConfig,
  getProfilesForExecution,
  getMergedOpportunitiesForAging,
} from "@/lib/shadowSimulationService";
import { getAllShadowProfiles, getRejectionCountsByProfile } from "@/lib/shadowSimulationStore";
import { getProfileById } from "@/lib/shadowSimulationProfiles";
import { getProfileEligibilityDiagnostics } from "@/lib/profileEligibilityFunnel";
import { getProfileObservabilityConsistency } from "@/lib/profileObservabilityConsistency";
import { getActiveTradeAgingDiagnostics } from "@/lib/activeTradeAgingDiagnostics";
import { buildProfileAtomicSnapshot } from "@/lib/profileAtomicSnapshot";

export const dynamic = "force-dynamic";

export async function GET(req: NextRequest) {
  try {
    ensureShadowSimulation();
    getProfilesForExecution();
    const rejectionCountsByProfile = getRejectionCountsByProfile();
    const currentOpportunitiesMap = await getMergedOpportunitiesForAging();
    const profiles = getAllShadowProfiles();

    const profileEligibilityDiagnostics = getProfileEligibilityDiagnostics(
      profiles.map((p) => p.profileId),
      (pid) => getProfileById(pid)?.label,
      (pid) => {
        const p = profiles.find((x) => x.profileId === pid);
        return p?.closedTrades.filter((t) => t.status === "closed" && t.closedAt).length ?? 0;
      },
      (pid) => {
        const p = profiles.find((x) => x.profileId === pid);
        return p?.realizedPnL ?? 0;
      },
      rejectionCountsByProfile
    );
    const profileObservabilityConsistency = getProfileObservabilityConsistency(
      profiles.map((p) => ({
        profileId: p.profileId,
        label: p.label,
        lastCycleProcessedAt: p.lastCycleProcessedAt,
        closedTradesCount: p.closedTrades.filter((t) => t.status === "closed" && t.closedAt).length,
        realizedPnL: p.realizedPnL ?? 0,
      })),
      profileEligibilityDiagnostics
    );
    const funnelByProfile: Record<string, { finalCandidateCount: number; openedTradeCount: number }> = {};
    for (const [pid, d] of Object.entries(profileEligibilityDiagnostics)) {
      funnelByProfile[pid] = {
        finalCandidateCount: d.funnel.finalCandidateCount,
        openedTradeCount: d.funnel.openedTradeCount,
      };
    }
    const activeTradeAgingDiagnostics = getActiveTradeAgingDiagnostics(
      profiles.map((p) => ({
        profileId: p.profileId,
        label: p.label,
        activeTrades: [...(p.activeTrades ?? [])],
        closedTradesCount: p.closedTrades.filter((t) => t.status === "closed" && t.closedAt).length,
      })),
      (pid) => getProfileById(pid) ?? getProfileConfig(pid) ?? undefined,
      currentOpportunitiesMap,
      funnelByProfile
    );

    const snapshot = buildProfileAtomicSnapshot(
      profiles.map((p) => p.profileId),
      (pid) => profileObservabilityConsistency.byProfile[pid] ?? null,
      (pid) => profileEligibilityDiagnostics[pid] ?? null,
      (pid) => activeTradeAgingDiagnostics.byProfile[pid] ?? null,
      (pid) => profiles.find((p) => p.profileId === pid)?.lastCycleProcessedAt ?? null,
      (pid) => profiles.find((p) => p.profileId === pid)?.label ?? getProfileById(pid)?.label
    );

    const profileId = req.nextUrl.searchParams.get("profileId");
    if (profileId) {
      const entry = snapshot.byProfile[profileId];
      if (!entry) {
        return NextResponse.json(
          { error: "Profile not found", profileId, available: Object.keys(snapshot.byProfile) },
          { status: 404 }
        );
      }
      return NextResponse.json({
        ...entry,
        generatedAt: snapshot.generatedAt,
      });
    }

    return NextResponse.json(snapshot);
  } catch (err) {
    console.error("[API /shadow/profile-snapshot]", err);
    return NextResponse.json(
      { error: "Profile snapshot failed", timestamp: new Date().toISOString() },
      { status: 500 }
    );
  }
}
