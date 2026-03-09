/**
 * Version / deploy verification — returns Railway build metadata.
 * Used to detect stale deployments. No business logic.
 */

import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

export async function GET() {
  const gitSha = process.env.RAILWAY_GIT_COMMIT_SHA ?? null;
  const domain = process.env.RAILWAY_PUBLIC_DOMAIN ?? null;
  const url = domain ? `https://${domain}` : null;

  return NextResponse.json({
    success: true,
    gitCommitSha: gitSha,
    gitCommitShaShort: gitSha ? gitSha.slice(0, 7) : null,
    dashboardUrl: url,
    timestamp: new Date().toISOString(),
    routeExists: true, // proves this route file was deployed
  });
}
