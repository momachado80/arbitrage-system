/**
 * Microcapital Readiness Gate — internal GET-only endpoint.
 *
 * Returns the dossier as JSON. Strict policy: shadow_only_no_api_submit_v1.
 *
 * Rules:
 *   - GET only.
 *   - Reads only — does not mutate state.
 *   - Does not start dispatcher, worker, or any external call.
 *   - Does not accept dangerous parameters (e.g. modes that would imply real submission).
 *   - apiSubmitAllowed remains false; no signer / wallet / private key / transaction is touched.
 *
 * The endpoint is paper/shadow only.
 */

import { NextResponse } from "next/server";

import { composeMicrocapitalReadinessDossier } from "@/lib/readiness/composeMicrocapitalReadinessDossier";

export const dynamic = "force-dynamic";

export async function GET(): Promise<NextResponse> {
  try {
    const dossier = composeMicrocapitalReadinessDossier();
    return NextResponse.json({
      ok: true,
      generatedAt: dossier.systemIdentity.generatedAt,
      mode: "shadow_only",
      readiness: {
        finalVerdict: dossier.finalVerdict,
        blockers: dossier.blockers,
        warnings: dossier.warnings,
        nextActions: dossier.nextActions,
        policyTag: dossier.policyTag,
        systemIdentity: dossier.systemIdentity,
        safetyReadiness: dossier.safetyStatus,
        economicReadiness: dossier.economicReadiness,
        executionRealism: dossier.executionRealism,
        reliabilityReadiness: dossier.reliabilityReadiness,
        observabilityReadiness: dossier.observabilityReadiness,
        riskLimitsSimulated: dossier.riskLimitsSimulated,
      },
    });
  } catch (err) {
    console.error("[API /readiness/microcapital]", err);
    return NextResponse.json(
      {
        ok: false,
        error: err instanceof Error ? err.message : String(err),
        mode: "shadow_only",
      },
      { status: 200 },
    );
  }
}
